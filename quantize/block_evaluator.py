from typing import Dict

import torch

from quantize.utils import (
    call_layer_forward,
    capture_router_labels_layerwise,
    clear_temp_variable,
    compute_topk_kl_loss,
    compute_topk_mse_loss,
    extract_hidden_states,
    set_quant_state,
    smooth_and_quant_temporary,
)


def _iter_batch_indices(total: int, batch_size: int):
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield start, end


def _fmt_metric(value: float) -> str:
    # Keep compact output and automatically switch to scientific notation for tiny values.
    return f"{float(value):.6g}"


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

    if teacher_indices is None or teacher_indices.dim() != 3:
        if logger is not None:
            logger.warning(
                "[BlockUpdate] Teacher router indices are unavailable or not 3D; skip auxiliary router loss"
            )
        return None

    if router_logits.dim() == 3:
        if router_logits.shape[:2] != teacher_indices.shape[:2]:
            if logger is not None:
                logger.warning(
                    "[BlockUpdate] Router logits shape mismatch: student=%s teacher=%s; skip auxiliary router loss",
                    tuple(router_logits.shape),
                    tuple(teacher_indices.shape),
                )
            return None
        return router_logits

    if router_logits.dim() != 2:
        if logger is not None:
            logger.warning(
                "[BlockUpdate] Unsupported router logits rank %s with shape=%s; skip auxiliary router loss",
                router_logits.dim(),
                tuple(router_logits.shape),
            )
        return None

    batch_size, seq_len, _ = teacher_indices.shape
    flat_tokens, num_experts = router_logits.shape

    if batch_size > 0 and flat_tokens == batch_size * seq_len:
        return router_logits.reshape(batch_size, seq_len, num_experts)

    if logger is not None:
        logger.warning(
            "[BlockUpdate] Router logits shape mismatch: student=%s teacher=%s; skip auxiliary router loss",
            tuple(router_logits.shape),
            tuple(teacher_indices.shape),
        )
    return None


def evaluate_block_loss_modes(
    qlayer,
    args,
    loss_func,
    quant_inputs,
    fp_targets,
    fp_targets_aug,
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
    Evaluate block-wise loss with student routing and return scalar logs.
    """
    results: Dict[str, Dict[str, float]] = {}
    set_quant_state(qlayer, weight_quant=False, act_quant=True)

    main_items = []
    aug_items = []
    total_items = []

    with torch.no_grad():
        for start, end in _iter_batch_indices(args.nsamples, args.batch_size):
            batch_attention_mask = _get_batch_attention_mask(attention_mask_batch, start, end)

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
                clear_temp_variable(qlayer)

            main_items.append(main_loss.detach().cpu())
            aug_items.append(aug_loss.detach().cpu())
            total_items.append(total_loss.detach().cpu())

    main_mean = torch.stack(main_items).mean().item()
    aug_mean = torch.stack(aug_items).mean().item() if aug_items else 0.0
    total_mean = torch.stack(total_items).mean().item()
    results["student"] = {
        "main": main_mean,
        "aug": aug_mean,
        "total": total_mean,
    }

    logger.info(
        f"[BlockEval] layer {layer_idx} epoch {epoch_idx} mode=student "
        f"main_loss:{_fmt_metric(main_mean)} aug_loss:{_fmt_metric(aug_mean)} total_loss:{_fmt_metric(total_mean)}"
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
    clip_grad=None,
    aux_use_kl=True,
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
        skipped_batches = 0
        skipped_aux_missing_teacher = 0
        skipped_aux_missing_router = 0
        skipped_aux_missing_alignment = 0
        skipped_aux_invalid_loss = 0

        for start, end in _iter_batch_indices(args.nsamples, args.batch_size):
            batch_attention_mask = _get_batch_attention_mask(attention_mask_batch, start, end)
            optimizer.zero_grad()
            norm = None
            skipped_update = False
            skip_reason = None

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
                    if aux_enabled:
                        if teacher_router_labels is None:
                            skipped_aux_missing_teacher += 1
                        elif router_logits is None:
                            skipped_aux_missing_router += 1
                        else:
                            teacher_logits = teacher_router_labels[0][start:end, :, :aux_topk]
                            teacher_indices = teacher_router_labels[1][start:end, :, :aux_topk]
                            aligned_router_logits = _align_router_logits_for_teacher(router_logits, teacher_indices, logger=logger)
                            if aligned_router_logits is None:
                                skipped_aux_missing_alignment += 1
                            else:
                                aux_loss_fn = compute_topk_kl_loss if aux_use_kl else compute_topk_mse_loss
                                aux_loss_value = aux_loss_fn(
                                    aligned_router_logits.float(),
                                    teacher_logits,
                                    teacher_indices,
                                    return_none_on_nonfinite=True,
                                )
                                if aux_loss_value is None:
                                    skipped_aux_invalid_loss += 1
                                else:
                                    aux_loss = aux_loss_value

                    # Scale the auxiliary router loss relative to the block
                    # reconstruction loss. main_loss spans several orders of
                    # magnitude across layers (~1e-5 shallow to ~0.6 deep) while
                    # the router KL/MSE lives in an unrelated space, so a fixed
                    # raw weight cannot stay balanced. We rescale aux so its
                    # numeric contribution tracks aux_weight*main, then clamp the
                    # scale at 1.0 so aux can never dominate main and a tiny aux
                    # (already-aligned router) cannot blow up its gradient.
                    aux_term = aux_loss
                    if aux_enabled and aux_weight > 0:
                        aux_denom = aux_loss.detach().clamp_min(1e-8)
                        scale = (main_loss.detach() / aux_denom).clamp(max=1.0)
                        aux_term = aux_loss * scale
                    total_loss = main_loss + aug_loss + (aux_weight * aux_term)

                norm, skipped_update, skip_reason = loss_scaler(
                    total_loss,
                    optimizer,
                    clip_grad=clip_grad,
                    parameters=clip_parameters,
                    return_metadata=True,
                )
                if norm is None:
                    norm = torch.tensor(0.0, device=quant_inputs.device)
                norm = norm.cpu()
            finally:
                clear_temp_variable(qlayer)

            if skipped_update:
                skipped_batches += 1
                logger.warning(
                    f"[SkipBatch] BlockUpdate layer {layer_idx} epoch {update_epoch} batch {start // args.batch_size}: "
                    f"optimizer update skipped due to {skip_reason}"
                )
                continue

            if clip_grad is not None and float(norm.item()) > float(clip_grad):
                logger.info(
                    f"[GradClip] BlockUpdate layer {layer_idx} epoch {update_epoch} "
                    f"batch {start // args.batch_size}: grad_norm={_fmt_metric(norm.item())} > max_norm={_fmt_metric(clip_grad)}"
                )

            total_items.append(total_loss.detach().cpu())
            main_items.append(main_loss.detach().cpu())
            aug_items.append(aug_loss.detach().cpu())
            aux_items.append(aux_loss.detach().cpu())
            norm_items.append(norm)

        if not total_items:
            logger.warning(f"[BlockUpdate] layer {layer_idx} epoch {update_epoch}: all batches skipped")
            continue

        total_mean = torch.stack(total_items).mean().item()
        main_mean = torch.stack(main_items).mean().item()
        aug_mean = torch.stack(aug_items).mean().item()
        aux_mean = torch.stack(aux_items).mean().item() if aux_items else 0.0
        norm_mean = torch.stack(norm_items).mean().item()
        final_total = total_mean

        logger.info(
            f"[BlockUpdate] layer {layer_idx} epoch {update_epoch} "
            f"main_loss:{_fmt_metric(main_mean)} aug_loss:{_fmt_metric(aug_mean)} "
            f"aux_loss:{_fmt_metric(aux_mean)} total_loss:{_fmt_metric(total_mean)} norm:{_fmt_metric(norm_mean)} skipped:{skipped_batches}"
        )

        if aux_enabled and (skipped_aux_missing_teacher or skipped_aux_missing_router or skipped_aux_missing_alignment or skipped_aux_invalid_loss):
            logger.warning(
                f"[BlockUpdate] layer {layer_idx} epoch {update_epoch} auxiliary router loss skipped "
                f"for teacher_missing={skipped_aux_missing_teacher} batches, "
                f"router_missing={skipped_aux_missing_router} batches, "
                f"alignment_missing={skipped_aux_missing_alignment} batches, "
                f"invalid_loss={skipped_aux_invalid_loss} batches"
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

