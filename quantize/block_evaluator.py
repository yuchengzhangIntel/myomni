from typing import Dict

import torch

from quantize.utils import (
    call_layer_forward,
    capture_router_labels_layerwise,
    clear_temp_variable,
    compute_topk_mse_loss,
    extract_hidden_states,
    set_quant_state,
    smooth_and_quant_temporary,
)


def _iter_batch_indices(total: int, batch_size: int):
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield start, end


def compute_fp_block_targets(
    layer,
    qlayer,
    args,
    fp_inputs,
    quant_inputs,
    layer_kwargs,
    attention_mask,
    position_ids,
    traincast,
):
    """
    Build fixed block-level targets used by block-wise loss.

    Main target follows the historical logic: use teacher fp layer outputs.
    Optional aug target follows current code path: use qlayer (de-quantized) outputs on quant inputs.
    """
    fp_targets = torch.zeros_like(fp_inputs)
    fp_targets_aug = torch.zeros_like(quant_inputs) if args.aug_loss else None

    set_quant_state(qlayer, weight_quant=False, act_quant=False)
    with torch.no_grad():
        with traincast():
            for sample_idx in range(args.nsamples):
                fp_targets[sample_idx] = extract_hidden_states(
                    call_layer_forward(
                        layer,
                        fp_inputs[sample_idx].unsqueeze(0),
                        layer_kwargs=layer_kwargs,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )
                )
                if fp_targets_aug is not None:
                    fp_targets_aug[sample_idx] = extract_hidden_states(
                        call_layer_forward(
                            qlayer,
                            quant_inputs[sample_idx].unsqueeze(0),
                            layer_kwargs=layer_kwargs,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                        )
                    )

    return fp_targets, fp_targets_aug


def capture_teacher_router_labels(
    layer,
    fp_inputs,
    device,
    layer_kwargs,
    topk,
    logger,
):
    if topk <= 0:
        return None

    return capture_router_labels_layerwise(
        layer,
        fp_inputs,
        device,
        topk=topk,
        logger=logger,
        layer_kwargs=layer_kwargs,
    )


def _build_forced_router_logits(router_output: torch.Tensor, teacher_indices: torch.Tensor) -> torch.Tensor:
    if router_output.dim() not in (2, 3):
        return router_output

    forced_logits = torch.full_like(router_output, -10000.0)
    if teacher_indices.numel() == 0:
        return forced_logits

    if router_output.dim() == 2:
        forced_indices = teacher_indices.reshape(-1, teacher_indices.shape[-1]).to(router_output.device)
        k = min(forced_indices.shape[-1], router_output.shape[-1])
        if k <= 0:
            return forced_logits
        ranks = torch.arange(k, 0, -1, device=router_output.device, dtype=router_output.dtype)
        forced_logits.scatter_(-1, forced_indices[:, :k], ranks.unsqueeze(0).expand(forced_indices.shape[0], -1))
        return forced_logits

    forced_indices = teacher_indices.to(router_output.device)
    k = min(forced_indices.shape[-1], router_output.shape[-1])
    if k <= 0:
        return forced_logits

    ranks = torch.arange(k, 0, -1, device=router_output.device, dtype=router_output.dtype)
    forced_logits.scatter_(
        -1,
        forced_indices[..., :k],
        ranks.view(1, 1, -1).expand(forced_indices.shape[0], forced_indices.shape[1], -1),
    )
    return forced_logits


def _teacher_forcing_hook_factory(teacher_indices_batch: torch.Tensor):
    def _hook(_module, _inputs, output):
        return _build_forced_router_logits(output, teacher_indices_batch)

    return _hook


def _get_batch_attention_mask(attention_mask_batch, start: int, end: int):
    if attention_mask_batch is None:
        return None
    return attention_mask_batch[: end - start]


def _align_router_logits_for_teacher(router_logits: torch.Tensor, teacher_indices: torch.Tensor, logger=None):
    """
    Align router logits to [batch, seq, experts] to match teacher top-k labels.

    Some MoE implementations emit router logits as flattened [batch*seq, experts].
    """
    if router_logits is None:
        return None

    if router_logits.dim() == 3:
        return router_logits

    if router_logits.dim() != 2:
        return router_logits

    if teacher_indices is None or teacher_indices.dim() != 3:
        return router_logits.unsqueeze(0)

    batch_size, seq_len, _ = teacher_indices.shape
    flat_tokens, num_experts = router_logits.shape

    if batch_size > 0 and flat_tokens == batch_size * seq_len:
        return router_logits.reshape(batch_size, seq_len, num_experts)

    if logger is not None:
        logger.warning(
            "[BlockUpdate] Router logits shape mismatch: student=%s teacher=%s; fallback to unsqueeze(0)",
            tuple(router_logits.shape),
            tuple(teacher_indices.shape),
        )
    return router_logits.unsqueeze(0)


def evaluate_block_loss_modes(
    qlayer,
    args,
    loss_func,
    quant_inputs,
    fp_targets,
    fp_targets_aug,
    teacher_router_labels,
    teacher_forcing_topk,
    layer_kwargs,
    attention_mask_batch,
    position_ids,
    traincast,
    logger,
    layer_idx,
    epoch_idx,
    smooth_is_llama,
):
    """
    Evaluate block-wise loss in two modes and return scalar logs.
    """
    results: Dict[str, Dict[str, float]] = {}
    set_quant_state(qlayer, weight_quant=False, act_quant=True)

    eval_modes = ["student", "teacher_forcing"]
    for mode in eval_modes:
        if mode == "teacher_forcing" and teacher_router_labels is None:
            continue

        main_items = []
        aug_items = []
        total_items = []

        with torch.no_grad():
            for start, end in _iter_batch_indices(args.nsamples, args.batch_size):
                batch_attention_mask = _get_batch_attention_mask(attention_mask_batch, start, end)
                gate_hook = None
                if mode == "teacher_forcing":
                    teacher_indices = teacher_router_labels[1][start:end, :, :teacher_forcing_topk]
                    gate_hook = qlayer.mlp.gate.register_forward_hook(_teacher_forcing_hook_factory(teacher_indices))

                try:
                    with traincast():
                        smooth_and_quant_temporary(qlayer, args, smooth_is_llama)
                        quant_out = extract_hidden_states(
                            call_layer_forward(
                                qlayer,
                                quant_inputs[start:end],
                                layer_kwargs=layer_kwargs,
                                attention_mask=batch_attention_mask,
                                position_ids=position_ids,
                            )
                        )
                        main_loss = loss_func(fp_targets[start:end], quant_out)
                        aug_loss = torch.zeros_like(main_loss)
                        if fp_targets_aug is not None:
                            aug_loss = loss_func(fp_targets_aug[start:end], quant_out)
                        total_loss = main_loss + aug_loss
                finally:
                    if gate_hook is not None:
                        gate_hook.remove()
                    clear_temp_variable(qlayer)

                main_items.append(main_loss.detach().cpu())
                aug_items.append(aug_loss.detach().cpu())
                total_items.append(total_loss.detach().cpu())

        main_mean = torch.stack(main_items).mean().item()
        aug_mean = torch.stack(aug_items).mean().item() if aug_items else 0.0
        total_mean = torch.stack(total_items).mean().item()
        results[mode] = {
            "main": main_mean,
            "aug": aug_mean,
            "total": total_mean,
        }

        logger.info(
            f"[BlockEval] layer {layer_idx} epoch {epoch_idx} mode={mode} "
            f"main_loss:{main_mean:.6f} aug_loss:{aug_mean:.6f} total_loss:{total_mean:.6f}"
        )

    return results


def update_block_parameters_with_loss(
    qlayer,
    args,
    optimizer,
    loss_scaler,
    clip_parameters,
    loss_func,
    quant_inputs,
    fp_targets,
    fp_targets_aug,
    teacher_router_labels,
    aux_enabled,
    aux_weight,
    aux_topk,
    layer_kwargs,
    attention_mask_batch,
    position_ids,
    traincast,
    logger,
    layer_idx,
    smooth_is_llama,
    update_epochs,
):
    """
    Update selected parameters with block-wise loss.

    This function updates only parameters included in the provided optimizer,
    so caller controls attention/router/shared-gate update scopes.
    """
    set_quant_state(qlayer, weight_quant=False, act_quant=True)
    final_total = None

    for update_epoch in range(update_epochs):
        total_items = []
        main_items = []
        aug_items = []
        aux_items = []
        norm_items = []

        for start, end in _iter_batch_indices(args.nsamples, args.batch_size):
            batch_attention_mask = _get_batch_attention_mask(attention_mask_batch, start, end)
            optimizer.zero_grad()

            try:
                with traincast():
                    smooth_and_quant_temporary(qlayer, args, smooth_is_llama)
                    quant_out, router_logits = _forward_with_router_logits(
                        qlayer,
                        quant_inputs[start:end],
                        layer_kwargs=layer_kwargs,
                        attention_mask=batch_attention_mask,
                        position_ids=position_ids,
                    )
                    main_loss = loss_func(fp_targets[start:end], quant_out)
                    aug_loss = torch.zeros_like(main_loss)
                    if fp_targets_aug is not None:
                        aug_loss = loss_func(fp_targets_aug[start:end], quant_out)

                    aux_loss = torch.zeros_like(main_loss)
                    if aux_enabled and teacher_router_labels is not None and router_logits is not None:
                        teacher_logits = teacher_router_labels[0][start:end, :, :aux_topk]
                        teacher_indices = teacher_router_labels[1][start:end, :, :aux_topk]
                        router_logits = _align_router_logits_for_teacher(router_logits, teacher_indices, logger=logger)
                        aux_loss = compute_topk_mse_loss(router_logits.float(), teacher_logits, teacher_indices)

                    total_loss = main_loss + aug_loss + (aux_weight * aux_loss)

                norm = loss_scaler(total_loss, optimizer, parameters=clip_parameters).cpu()
            finally:
                clear_temp_variable(qlayer)

            total_items.append(total_loss.detach().cpu())
            main_items.append(main_loss.detach().cpu())
            aug_items.append(aug_loss.detach().cpu())
            aux_items.append(aux_loss.detach().cpu())
            norm_items.append(norm)

        total_mean = torch.stack(total_items).mean().item()
        main_mean = torch.stack(main_items).mean().item()
        aug_mean = torch.stack(aug_items).mean().item()
        aux_mean = torch.stack(aux_items).mean().item() if aux_items else 0.0
        norm_mean = torch.stack(norm_items).mean().item()
        final_total = total_mean

        logger.info(
            f"[BlockUpdate] layer {layer_idx} epoch {update_epoch} "
            f"main_loss:{main_mean:.6f} aug_loss:{aug_mean:.6f} "
            f"aux_loss:{aux_mean:.6f} total_loss:{total_mean:.6f} norm:{norm_mean:.6f}"
        )

    return final_total


def _forward_with_router_logits(layer, hidden_states, layer_kwargs=None, **kwargs):
    captured = {}

    def _hook(_module, _inputs, output):
        captured["logits"] = output

    handle = None
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
        handle = layer.mlp.gate.register_forward_hook(_hook)

    try:
        outputs = call_layer_forward(layer, hidden_states, layer_kwargs=layer_kwargs, **kwargs)
    finally:
        if handle is not None:
            handle.remove()

    return extract_hidden_states(outputs), captured.get("logits")


