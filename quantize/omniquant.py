import torch
import torch.nn as nn
import torch.nn.functional as F
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_falcon_layer import QuantFalconDecoderLayer
from quantize.int_linear import QuantLinear
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc
from quantize.utils import (
    let_parameters, lwc_parameters, get_omni_parameters,
    omni_state_dict, register_scales_and_zeros, smooth_and_quant_temporary,
    smooth_and_quant_inplace, clear_temp_variable, set_quant_state,
    capture_router_labels_layerwise, compute_expert_shift_detailed,
    compute_topk_mse_loss, forward_with_router_logits, create_router_hook,
    call_layer_forward, extract_hidden_states
)
from quantize.moe_utils import (
    QuantizedPackedExperts,
    compute_expert_down_proj_output,
    compute_router_scores,
    get_moe_experts_module,
    get_moe_top_k,
    get_shared_expert_module,
    is_packed_experts_module,
    pin_cpu_tensor,
    select_top_n_experts,
)


class LoraLinear(nn.Module):
    """
    LoRA wrapper for nn.Linear layer.
    Freezes the original weight and trains low-rank adapters lora_A and lora_B.
    """
    def __init__(self, linear: nn.Linear, r: int = 8, alpha: float = 16, seed: int = 42):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # Copy original weight and bias, freeze them
        self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter('bias', None)
        
        # Initialize LoRA matrices with fixed seed for reproducibility
        # A with Gaussian (scaled), B with zeros
        generator = torch.Generator(device=linear.weight.device)
        generator.manual_seed(seed)
        self.lora_A = nn.Parameter(torch.randn(r, self.in_features, device=linear.weight.device, dtype=linear.weight.dtype, generator=generator) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, device=linear.weight.device, dtype=linear.weight.dtype))
    
    def forward(self, x):
        # output = x @ (W + scaling * B @ A).T + bias
        # = x @ W.T + scaling * x @ A.T @ B.T + bias
        base_out = nn.functional.linear(x, self.weight, self.bias)
        lora_out = nn.functional.linear(nn.functional.linear(x, self.lora_A), self.lora_B)
        return base_out + self.scaling * lora_out
    
    def merge(self):
        """
        Merge LoRA weights into the original weight and return a standard nn.Linear.
        """
        # Merge: W_new = W + scaling * B @ A
        merged_weight = self.weight.data + self.scaling * (self.lora_B.data @ self.lora_A.data)
        
        linear = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None, 
                          device=merged_weight.device, dtype=merged_weight.dtype)
        linear.weight.data = merged_weight
        if self.bias is not None:
            linear.bias.data = self.bias.data.clone()
        
        return linear
try:
    import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
    import auto_gptq.nn_modules.qlinear.qlinear_triton as qlinear_triton
except:
    print("auto_gptq is required for real quantization")



def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}


def add_new_module(name, original_module, added_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)     


def collect_stage_parameters(module, prefixes, include_linear_lora=False):
    lwc_params = []
    linear_lora_params = []
    seen = set()

    for name, param in module.named_parameters():
        if not any(name.startswith(prefix) for prefix in prefixes):
            continue
        if "bound_factor" in name and id(param) not in seen:
            lwc_params.append(param)
            seen.add(id(param))
        elif include_linear_lora and (name.endswith("lora_A") or name.endswith("lora_B")) and id(param) not in seen:
            linear_lora_params.append(param)
            seen.add(id(param))

    return lwc_params, linear_lora_params


def compute_attention_outputs(layer, hidden_states, layer_kwargs, attention_mask=None, position_ids=None):
    normed_hidden_states = layer.input_layernorm(hidden_states)
    return extract_hidden_states(call_layer_forward(
        layer.self_attn,
        normed_hidden_states,
        layer_kwargs=layer_kwargs,
        attention_mask=attention_mask,
        position_ids=position_ids,
    ))


def compute_mlp_inputs(layer, hidden_states, layer_kwargs, attention_mask=None, position_ids=None):
    attention_outputs = compute_attention_outputs(
        layer,
        hidden_states,
        layer_kwargs,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    post_attention_hidden_states = hidden_states + attention_outputs
    mlp_inputs = layer.post_attention_layernorm(post_attention_hidden_states)
    return attention_outputs, post_attention_hidden_states, mlp_inputs


def resolve_quant_routing_top_n(moe_module, requested_top_n):
    layer_top_k = get_moe_top_k(moe_module)
    if layer_top_k is None:
        return requested_top_n
    if requested_top_n is None:
        return layer_top_k
    if requested_top_n < layer_top_k:
        raise ValueError(
            f"quant_routing_top_n ({requested_top_n}) must be greater than or equal to the layer top-k ({layer_top_k})"
        )
    return requested_top_n


def build_moe_label_cache(layer, fp_inputs, layer_kwargs, attention_mask, position_ids, top_n):
    label_cache = []
    shared_expert = get_shared_expert_module(layer.mlp)
    experts_module = get_moe_experts_module(layer.mlp)

    with torch.no_grad():
        for sample_idx in range(fp_inputs.shape[0]):
            sample_inputs = fp_inputs[sample_idx].unsqueeze(0)
            _, _, mlp_inputs = compute_mlp_inputs(
                layer,
                sample_inputs,
                layer_kwargs,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            flat_mlp_inputs = mlp_inputs.reshape(-1, mlp_inputs.shape[-1])
            router_scores = compute_router_scores(layer.mlp, flat_mlp_inputs)
            top_indices, top_weights = select_top_n_experts(router_scores, top_n)

            sample_cache = {
                "expert_labels": {},
                "shared_labels": None,
            }

            for expert_idx_tensor in torch.unique(top_indices):
                expert_idx = int(expert_idx_tensor.item())
                token_idx, top_pos = torch.where(top_indices == expert_idx)
                labels = compute_expert_down_proj_output(
                    experts_module,
                    expert_idx,
                    flat_mlp_inputs[token_idx],
                )
                sample_cache["expert_labels"][expert_idx] = {
                    "token_idx": pin_cpu_tensor(token_idx),
                    "weights": pin_cpu_tensor(top_weights[token_idx, top_pos]),
                    "labels": pin_cpu_tensor(labels),
                }

            if shared_expert is not None:
                shared_outputs = extract_hidden_states(call_layer_forward(shared_expert, flat_mlp_inputs))
                sample_cache["shared_labels"] = pin_cpu_tensor(shared_outputs)

            label_cache.append(sample_cache)

    return label_cache


def compute_moe_self_supervision_loss(
    qlayer,
    quant_inputs,
    batch_label_cache,
    layer_kwargs,
    attention_mask,
    position_ids,
    use_router_weight_in_loss,
    precomputed_mlp_inputs=None,
):
    if precomputed_mlp_inputs is None:
        _, _, mlp_inputs = compute_mlp_inputs(
            qlayer,
            quant_inputs,
            layer_kwargs,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
    else:
        mlp_inputs = precomputed_mlp_inputs
    experts_module = get_moe_experts_module(qlayer.mlp)
    shared_expert = get_shared_expert_module(qlayer.mlp)
    loss_terms = []

    for batch_idx, sample_cache in enumerate(batch_label_cache):
        sample_inputs = mlp_inputs[batch_idx]
        for expert_idx, cached_values in sample_cache["expert_labels"].items():
            token_idx = cached_values["token_idx"].to(sample_inputs.device, non_blocking=True)
            teacher_labels = cached_values["labels"].to(sample_inputs.device, non_blocking=True)
            student_labels = compute_expert_down_proj_output(
                experts_module,
                expert_idx,
                sample_inputs[token_idx],
            )
            expert_loss = (student_labels.float() - teacher_labels.float()).pow(2).mean(dim=-1)
            if use_router_weight_in_loss:
                weights = cached_values["weights"].to(sample_inputs.device, non_blocking=True).float()
                expert_loss = expert_loss * weights
            loss_terms.append(expert_loss)

        if sample_cache["shared_labels"] is not None and shared_expert is not None:
            shared_labels = sample_cache["shared_labels"].to(sample_inputs.device, non_blocking=True)
            student_shared = extract_hidden_states(call_layer_forward(shared_expert, sample_inputs))
            shared_loss = (student_shared.float() - shared_labels.float()).pow(2).mean(dim=-1)
            loss_terms.append(shared_loss)

    if not loss_terms:
        return torch.tensor(0.0, device=quant_inputs.device, requires_grad=True)

    return torch.cat([term.reshape(-1) for term in loss_terms]).mean()


def train_decoupled_moe_layer(
    layer,
    qlayer,
    args,
    logger,
    layer_idx,
    fp_inps,
    quant_inps,
    layer_kwargs,
    attention_mask_batch,
    attention_mask,
    position_ids,
    traincast,
    quant_routing_top_n,
    use_router_weight_in_loss,
):
    if args.let:
        raise ValueError("Decoupled Qwen/DeepSeek MoE training does not support --let")
    if getattr(args, "train_gate_lora", False):
        raise ValueError("Decoupled Qwen/DeepSeek MoE training does not support --train_gate_lora without an explicit router loss")
    if getattr(args, "train_shared_gate", False):
        raise ValueError("Decoupled Qwen/DeepSeek MoE training does not support --train_shared_gate without an explicit shared-gate loss")
    if getattr(args, "calibrate_router", False):
        raise ValueError("Decoupled Qwen/DeepSeek MoE training does not support --calibrate_router")

    qlayer.float()
    final_stage_loss = None
    loss_func = torch.nn.MSELoss()
    attention_prefixes = ("self_attn.",)
    moe_prefixes = ("mlp.experts.", "mlp.shared_expert.", "mlp.shared_experts.")

    attn_lwc_params, attn_lora_params = collect_stage_parameters(
        qlayer,
        attention_prefixes,
        include_linear_lora=getattr(args, "use_linear_lora", False),
    )
    attn_param_groups = []
    if attn_lwc_params:
        attn_param_groups.append({"params": attn_lwc_params, "lr": args.lwc_lr, "weight_decay": 0})
    if attn_lora_params:
        attn_param_groups.append({"params": attn_lora_params, "lr": args.linear_lora_lr, "weight_decay": args.wd})

    if attn_param_groups and args.epochs > 0:
        logger.info(f"[Decoupled Attention] Layer {layer_idx}: training attention quantization for {args.epochs} epochs")
        attn_optimizer = torch.optim.AdamW(attn_param_groups, weight_decay=0)
        attn_scaler = utils.NativeScalerWithGradNormCount()
        attn_clip_params = attn_lwc_params + attn_lora_params
        for epoch in range(args.epochs):
            loss_list = []
            norm_list = []
            for start in range(0, args.nsamples, args.batch_size):
                end = min(start + args.batch_size, args.nsamples)
                batch_attention_mask = attention_mask_batch[: end - start] if attention_mask_batch is not None else None
                attn_optimizer.zero_grad()
                with torch.no_grad():
                    with torch.amp.autocast('cuda'):
                        teacher_attn_outputs = compute_attention_outputs(
                            layer,
                            fp_inps[start:end],
                            layer_kwargs,
                            attention_mask=batch_attention_mask,
                            position_ids=position_ids,
                        )
                with traincast():
                    smooth_and_quant_temporary(qlayer, args, isllama=True)
                    student_attn_outputs = compute_attention_outputs(
                        qlayer,
                        quant_inps[start:end],
                        layer_kwargs,
                        attention_mask=batch_attention_mask,
                        position_ids=position_ids,
                    )
                    loss = loss_func(student_attn_outputs, teacher_attn_outputs)
                loss_list.append(loss.detach().cpu())
                norm = attn_scaler(loss, attn_optimizer, parameters=attn_clip_params).cpu()
                norm_list.append(norm)
                clear_temp_variable(qlayer)

            loss_mean = torch.stack(loss_list).mean()
            norm_mean = torch.stack(norm_list).mean()
            logger.info(f"[Decoupled Attention] Layer {layer_idx} epoch {epoch} loss:{loss_mean} norm:{norm_mean}")
            final_stage_loss = loss_mean.item()
        del attn_optimizer

    top_n = resolve_quant_routing_top_n(layer.mlp, quant_routing_top_n)
    logger.info(f"[Decoupled MoE] Layer {layer_idx}: building CPU label cache with top_n={top_n}")
    label_cache = build_moe_label_cache(
        layer,
        fp_inps,
        layer_kwargs,
        attention_mask,
        position_ids,
        top_n,
    )

    moe_lwc_params, moe_lora_params = collect_stage_parameters(
        qlayer,
        moe_prefixes,
        include_linear_lora=getattr(args, "use_linear_lora", False),
    )
    moe_param_groups = []
    if moe_lwc_params:
        moe_param_groups.append({"params": moe_lwc_params, "lr": args.lwc_lr, "weight_decay": 0})
    if moe_lora_params:
        moe_param_groups.append({"params": moe_lora_params, "lr": args.linear_lora_lr, "weight_decay": args.wd})

    if moe_param_groups and args.epochs > 0:
        logger.info(f"[Decoupled MoE] Layer {layer_idx}: training expert self-supervision for {args.epochs} epochs")
        moe_optimizer = torch.optim.AdamW(moe_param_groups, weight_decay=0)
        moe_scaler = utils.NativeScalerWithGradNormCount()
        moe_clip_params = moe_lwc_params + moe_lora_params
        for epoch in range(args.epochs):
            loss_list = []
            norm_list = []
            for start in range(0, args.nsamples, args.batch_size):
                end = min(start + args.batch_size, args.nsamples)
                batch_attention_mask = attention_mask_batch[: end - start] if attention_mask_batch is not None else None
                moe_optimizer.zero_grad()
                with traincast():
                    smooth_and_quant_temporary(qlayer, args, isllama=True)
                with torch.no_grad():
                    with traincast():
                        _, _, detached_mlp_inputs = compute_mlp_inputs(
                            qlayer,
                            quant_inps[start:end],
                            layer_kwargs,
                            attention_mask=batch_attention_mask,
                            position_ids=position_ids,
                        )
                        detached_mlp_inputs = detached_mlp_inputs.detach()
                with traincast():
                    loss = compute_moe_self_supervision_loss(
                        qlayer,
                        quant_inps[start:end],
                        label_cache[start:end],
                        layer_kwargs,
                        batch_attention_mask,
                        position_ids,
                        use_router_weight_in_loss,
                        precomputed_mlp_inputs=detached_mlp_inputs,
                    )
                loss_list.append(loss.detach().cpu())
                norm = moe_scaler(loss, moe_optimizer, parameters=moe_clip_params).cpu()
                norm_list.append(norm)
                clear_temp_variable(qlayer)

            loss_mean = torch.stack(loss_list).mean()
            norm_mean = torch.stack(norm_list).mean()
            logger.info(f"[Decoupled MoE] Layer {layer_idx} epoch {epoch} loss:{loss_mean} norm:{norm_mean}")
            final_stage_loss = loss_mean.item()
        del moe_optimizer

    del label_cache
    return final_stage_loss

def omniquant(
    lm,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
    train_shared_gate=False,
    train_gate_lora=False,
    shared_gate_lr=1e-4,
    gate_lora_lr=1e-4,
    lora_r=8,
    lora_alpha=16,
    # Router Calibration parameters
    calibrate_router=False,
    router_lr=1e-3,
    router_epochs=5,
    k_loss=20,      # TopK for loss calculation (cached label size)
    k_routing=4,    # TopK for expert shift metric (actual routing k)
    quant_routing_top_n=None,
    use_router_weight_in_loss=False,
):
    logger.info("Starting ...")
    
    # WandB integration: import and initialize global step for continuous training curve
    wandb = None
    if getattr(args, 'enable_wandb', False):
        try:
            import wandb as _wandb
            wandb = _wandb
        except ImportError:
            logger.warning("WandB not installed but enable_wandb=True. Skipping WandB logging.")
    global_step = 0  # Global step counter for continuous WandB logging across layers
    final_loss = None  # Track the loss from the last epoch of the last layer
    expert_shift_data = []  # Collect expert shift data per layer for visualization
    
    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    if "llama" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1"
        }
        layer_name_prefix = "model.layers"
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "out_proj":"out",
            "fc1":"fc1"
        }
        layer_name_prefix = "model.decoder.layers"
    elif "falcon" in args.net.lower():
        layers = model.transformer.h
        model.transformer.word_embeddings.to(dev)
        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)
        DecoderLayer = QuantFalconDecoderLayer
        layer_name_prefix = "model.transformer.h"
    elif 'mixtral' in args.net.lower():
        is_llama = True   # same to llama except ffn
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        layer_name_prefix = "model.layers"
    elif 'qwen' in args.net.lower() or 'deepseek' in args.net.lower():
        is_llama = True   # same to llama except ffn (MoE structure)
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        # Qwen/DeepSeek MoE models only support the MoE/LWC path here, no DecoderLayer wrapper is needed.
        layer_name_prefix = "model.layers"
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral/qwen/deepseek now")

    if ("qwen" in args.net.lower() or "deepseek" in args.net.lower()) and args.let:
        raise ValueError("Decoupled Qwen/DeepSeek MoE training does not support --let")
    
    
    layers[0] = layers[0].to(dev)
    if args.deactive_amp and args.epochs>0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = lambda: torch.amp.autocast('cuda')
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["layer_kwargs"] = dict(kwargs)
            cache["attention_mask"] = kwargs.get("attention_mask")
            if self.is_llama:
                cache["position_ids"] = kwargs.get("position_ids")
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "llama" in args.net.lower() or "mixtral" in args.net.lower() or "qwen" in args.net.lower() or "deepseek" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "opt" in args.net.lower():
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif 'falcon' in args.model:
        model.transformer.word_embeddings =  model.transformer.word_embeddings.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral/qwen/deepseek now")
    torch.cuda.empty_cache()

    
    # same input of first layer for fp model and quant model
    quant_inps = inps
    fp_inps = copy.deepcopy(inps)   # take output of fp model as input
    fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None # take output of quantization model as input
    
    attention_mask = cache["attention_mask"]

    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None

    loss_func = torch.nn.MSELoss()
    layer_kwargs = dict(cache.get("layer_kwargs", {}))
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None



    if args.resume:
        omni_parameters = torch.load(args.resume, weights_only=False)
    else:
        omni_parameters = {}

    
    
    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)
        current_fp_layer_inputs = fp_inps
        if "mixtral" in args.net.lower() or "qwen" in args.net.lower() or "deepseek" in args.net.lower():  
            # For MoE models (Mixtral, Qwen/DeepSeek MoE), only the LWC-style path is supported.
            # Simply replace Linear with QuantLinear, do not quantize router (gate)
            qlayer = copy.deepcopy(layer)

            for name, module in list(qlayer.named_modules()):
                if is_packed_experts_module(module):
                    add_new_module(name, qlayer, QuantizedPackedExperts(module, args))
            
            for name, module in list(qlayer.named_modules()):
                if isinstance(module, torch.nn.Linear):
                    # Target 1: Shared Expert Gate - name ends with "shared_expert_gate"
                    is_shared_expert_gate = name.endswith("shared_expert_gate")
                    
                    # Target 2: Router Gate - name ends with ".gate" (NOT "gate_proj")
                    # e.g., "mlp.gate" is router, but "mlp.experts.0.gate_proj" is NOT
                    is_router_gate = name.endswith(".gate") or name == "gate"
                    
                    if is_shared_expert_gate:
                        if train_shared_gate:
                            # Keep as nn.Linear but make trainable
                            module.weight.requires_grad = True
                            if module.bias is not None:
                                module.bias.requires_grad = True
                        # else: skip, keep as frozen nn.Linear (default behavior)
                    elif is_router_gate:
                        if train_gate_lora:
                            # Replace with LoraLinear wrapper (use layer index as seed for reproducibility)
                            lora_linear = LoraLinear(module, r=lora_r, alpha=lora_alpha, seed=args.seed + i)
                            add_new_module(name, qlayer, lora_linear)
                        # else: skip, keep as frozen nn.Linear (default behavior)
                    else:
                        # Target 3: All other linear layers (including gate_proj)
                        # Replace with QuantLinear
                        is_attn_linear = name.startswith("self_attn.") and name.split(".")[-1] in {"q_proj", "k_proj", "v_proj", "o_proj"}
                        weight_params = args.attn_weight_quant_params if (is_attn_linear and args.attn_weight_quant_params is not None) else args.weight_quant_params
                        quantlinear = QuantLinear(
                            module,
                            weight_params,
                            args.act_quant_params,
                            use_linear_lora=getattr(args, 'use_linear_lora', False),
                            linear_lora_r=getattr(args, 'linear_lora_r', 16),
                            linear_lora_alpha=getattr(args, 'linear_lora_alpha', 16.0),
                        )
                        add_new_module(name, qlayer, quantlinear)    
        else:
            qlayer = DecoderLayer(lm.model.config, layer, args)
        qlayer = qlayer.to(dev)

        use_decoupled_moe_training = (
            ("qwen" in args.net.lower() or "deepseek" in args.net.lower())
            and hasattr(layer, "mlp")
            and get_moe_experts_module(layer.mlp) is not None
        )

        # =================================================================
        # Legacy Expert Shift Tracking for Router Calibration
        # This path is kept for the older router-focused Qwen calibration flow.
        # =================================================================
        is_qwen_moe = "qwen" in args.net.lower() and not use_decoupled_moe_training
        cached_router_labels = None
        pre_lwc_shift = None
        post_calib_shift = None
        
        if is_qwen_moe:
            logger.info(f"[Expert Shift] Layer {i}: Starting expert shift tracking...")
            
            # Convert qlayer to FP32 for stable computation (same as LWC training)
            # This is required because:
            # 1. V100 doesn't support BF16, and autocast converts to FP16 which can cause precision issues
            # 2. GradScaler doesn't support FP16 gradients
            # 3. LWC training also uses qlayer.float() before training
            qlayer.float()
            
            # ============================================================
            # Phase A: Capture FP16 router labels from ORIGINAL layer
            # ============================================================
            logger.info(f"[Expert Shift] Layer {i}: Capturing FP16 router labels (topk={k_loss})...")
            cached_router_labels = capture_router_labels_layerwise(
                layer, fp_inps, dev, topk=k_loss, logger=logger, layer_kwargs=layer_kwargs
            )
            
            if cached_router_labels is not None:
                teacher_logits, teacher_indices = cached_router_labels
                logger.info(f"[Expert Shift] Layer {i}: teacher_logits shape={teacher_logits.shape}, teacher_indices shape={teacher_indices.shape}")
                seqlen = fp_inps.shape[1]
                
                # ============================================================
                # Phase B: Set quantization state and compute Pre-LWC Shift
                # ============================================================
                logger.info(f"[Expert Shift] Layer {i}: Setting quantization state...")
                
                # Set quantization state BEFORE computing expert shift
                # weight_quant=True: use quantized weights (temp_weight) to measure real quantization impact
                # act_quant=True: use quantized activations
                # This ensures we measure the true impact of quantization on router
                set_quant_state(qlayer, weight_quant=True, act_quant=True)
                
                # Create temp_weight ONCE for all samples (weights don't change during eval)
                smooth_and_quant_temporary(qlayer, args, is_llama)
                
                # Helper function to compute expert shift (temp_weight already set)
                def compute_shift_metrics(layer, inputs, teacher_idx, num_samples=8, desc=""):
                    """
                    Compute expert shift metrics using hook mechanism.
                    Assumes temp_weight is already set on the layer.
                    """
                    shift_any_sum = 0.0
                    shift_half_sum = 0.0
                    shift_all_sum = 0.0
                    for j in range(min(num_samples, inputs.shape[0])):
                        # Use hook-based forward to get router logits
                        # Use autocast to handle dtype mismatch (input FP16, weights FP32)
                        with torch.amp.autocast('cuda'):
                            out, router_logits = forward_with_router_logits(
                                layer,
                                inputs[j].unsqueeze(0),
                                layer_kwargs=layer_kwargs,
                                attention_mask=attention_mask,
                                position_ids=position_ids
                            )
                        
                        if router_logits is not None:
                            if router_logits.dim() == 2:
                                router_logits = router_logits.unsqueeze(0)
                            
                            if teacher_idx.dim() == 2:
                                t_idx = teacher_idx[j*seqlen:(j+1)*seqlen].unsqueeze(0)
                            else:
                                t_idx = teacher_idx[j:j+1]
                            
                            metrics = compute_expert_shift_detailed(
                                router_logits, t_idx, k_routing, debug=(j == 0 and desc != "")
                            )
                            shift_any_sum += metrics["shift_any"]
                            shift_half_sum += metrics["shift_half"]
                            shift_all_sum += metrics["shift_all"]
                    
                    n = min(num_samples, inputs.shape[0])
                    return shift_any_sum / n, shift_half_sum / n, shift_all_sum / n
                
                # Compute Pre-LWC Expert Shift (temp_weight already set)
                with torch.no_grad():
                    pre_shift_any, pre_shift_half, pre_shift_all = compute_shift_metrics(
                        qlayer, quant_inps, teacher_indices, 
                        num_samples=8, desc="Pre-LWC"
                    )
                    logger.info(f"[Expert Shift] Layer {i}: Pre-LWC Expert Shift (Quantized vs FP) - Any: {pre_shift_any:.4f}, Half: {pre_shift_half:.4f}, All: {pre_shift_all:.4f}")
                    pre_lwc_shift = (pre_shift_any, pre_shift_half, pre_shift_all)
                
                # ============================================================
                # Phase C: Router Calibration Training (Optional)
                # ============================================================
                # Only executed when calibrate_router=True
                # Goal: Train router to mimic FP16 expert selection UNDER QUANTIZED CONDITIONS
                # - Keep quantization enabled so router sees quantized hidden states
                # - temp_weight is already set from Phase B, no need to recreate
                if calibrate_router:
                    logger.info(f"[Router Calibration] Layer {i}: Router calibration training (epochs={router_epochs}, lr={router_lr})...")
                
                    # Freeze ALL parameters except router gate
                    saved_requires_grad = {}
                    for name, param in qlayer.named_parameters():
                        saved_requires_grad[name] = param.requires_grad
                        param.requires_grad = False
                    
                    # Enable gradient only for router gate (directly on qlayer, not wrapped)
                    # Support both nn.Linear and LoraLinear (when train_gate_lora is enabled)
                    router_gate_params = []
                    seen_params = set()
                    router_gate_module = None
                    for name, module in qlayer.named_modules():
                        if name.endswith("mlp.gate"):
                            router_gate_module = module
                            if isinstance(module, LoraLinear):
                                # For LoraLinear: train the original weight only, not LoRA matrices
                                # Temporarily enable gradient for the frozen weight
                                if id(module.weight) not in seen_params:
                                    module.weight.requires_grad = True
                                    router_gate_params.append(module.weight)
                                    seen_params.add(id(module.weight))
                                    if module.bias is not None and id(module.bias) not in seen_params:
                                        module.bias.requires_grad = True
                                        router_gate_params.append(module.bias)
                                        seen_params.add(id(module.bias))
                                    logger.info(f"[Router Calibration] Layer {i}: Enabled gradient for {name} (LoraLinear.weight)")
                            elif isinstance(module, nn.Linear):
                                if id(module.weight) not in seen_params:
                                    module.weight.requires_grad = True
                                    router_gate_params.append(module.weight)
                                    seen_params.add(id(module.weight))
                                    if module.bias is not None and id(module.bias) not in seen_params:
                                        module.bias.requires_grad = True
                                        router_gate_params.append(module.bias)
                                        seen_params.add(id(module.bias))
                                    logger.info(f"[Router Calibration] Layer {i}: Enabled gradient for {name} (nn.Linear)")
                    
                    if router_gate_params:
                        logger.info(f"[Router Calibration] Layer {i}: {len(router_gate_params)} unique parameters to optimize")
                        
                        # qlayer is already FP32 (converted at start of expert shift tracking)
                        router_optimizer = torch.optim.AdamW(router_gate_params, lr=router_lr, weight_decay=0)
                        
                        # Setup hook once for all training iterations using simple closure
                        hook_fn, get_logits, clear_logits = create_router_hook()
                        hook_handle = qlayer.mlp.gate.register_forward_hook(hook_fn)
                        
                        for epoch in range(router_epochs):
                            epoch_loss = 0.0
                            valid_samples = 0
                            for j in range(args.nsamples):
                                router_optimizer.zero_grad()
                                clear_logits()
                                
                                # Forward pass - qlayer is FP32, use autocast for efficiency
                                # Must call smooth_and_quant_temporary each iteration to recreate computation graph
                                with torch.amp.autocast('cuda'):
                                    smooth_and_quant_temporary(qlayer, args, is_llama)
                                    _ = call_layer_forward(
                                        qlayer,
                                        quant_inps[j].unsqueeze(0),
                                        layer_kwargs=layer_kwargs,
                                        attention_mask=attention_mask,
                                        position_ids=position_ids
                                    )
                                router_logits = get_logits()
                                
                                if router_logits is not None:
                                    if router_logits.dim() == 2:
                                        router_logits = router_logits.unsqueeze(0)
                                    
                                    if teacher_indices.dim() == 2:
                                        teacher_logit_sample = teacher_logits[j*seqlen:(j+1)*seqlen].unsqueeze(0)
                                        teacher_idx_sample = teacher_indices[j*seqlen:(j+1)*seqlen].unsqueeze(0)
                                    else:
                                        teacher_logit_sample = teacher_logits[j:j+1]
                                        teacher_idx_sample = teacher_indices[j:j+1]
                                    
                                    # Compute loss in FP32 for numerical stability
                                    loss = compute_topk_mse_loss(
                                        router_logits.float(),
                                        teacher_logit_sample,
                                        teacher_idx_sample,
                                        debug=(j == 0 and epoch == 0)
                                    )
                                    
                                    if not (torch.isnan(loss) or torch.isinf(loss)):
                                        loss.backward()
                                        
                                        # Clip gradients to prevent explosion
                                        torch.nn.utils.clip_grad_norm_(router_gate_params, max_norm=1.0)
                                        
                                        # Check for NaN gradients before stepping
                                        has_nan_grad = any(p.grad is not None and torch.isnan(p.grad).any() for p in router_gate_params)
                                        if not has_nan_grad:
                                            router_optimizer.step()
                                            epoch_loss += loss.item()
                                            valid_samples += 1
                            
                            avg_loss = epoch_loss / max(valid_samples, 1)
                            logger.info(f"[Router Calibration] Layer {i} Epoch {epoch}: TopK-MSE Loss = {avg_loss:.6f} ({valid_samples}/{args.nsamples} valid)")
                        
                        hook_handle.remove()
                        del router_optimizer
                    
                    # Restore requires_grad states
                    for name, param in qlayer.named_parameters():
                        if name in saved_requires_grad:
                            param.requires_grad = saved_requires_grad[name]
                    
                    # ============================================================
                    # Phase D: Compute Post-Calibration Expert Shift
                    # temp_weight is still set from Phase B, reuse it
                    # ============================================================
                    with torch.no_grad():
                        post_shift_any, post_shift_half, post_shift_all = compute_shift_metrics(
                            qlayer, quant_inps, teacher_indices, 
                            num_samples=8, desc=""
                        )
                        logger.info(f"[Router Calibration] Layer {i}: Post-Calib Expert Shift - Any: {post_shift_any:.4f}, Half: {post_shift_half:.4f}, All: {post_shift_all:.4f}")
                        logger.info(f"[Router Calibration] Layer {i}: Router Calib Improvement - Any: {pre_shift_any - post_shift_any:.4f}, Half: {pre_shift_half - post_shift_half:.4f}, All: {pre_shift_all - post_shift_all:.4f}")
                        post_calib_shift = (post_shift_any, post_shift_half, post_shift_all)
                    
                    logger.info(f"[Router Calibration] Layer {i}: Router calibration complete. Continuing to LWC training...")
                
                # Clean up temp_weight after expert shift tracking / router calibration
                clear_temp_variable(qlayer)
            else:
                logger.warning(f"[Expert Shift] Layer {i}: No router gate found, skipping expert shift tracking.")
        
        # obtain output of full-precision model
        set_quant_state(qlayer, weight_quant=False, act_quant=False)
        if args.epochs > 0 and not use_decoupled_moe_training:
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    for j in range(args.nsamples):
                        fp_inps[j] = extract_hidden_states(call_layer_forward(
                            qlayer,
                            fp_inps[j].unsqueeze(0),
                            layer_kwargs=layer_kwargs,
                            attention_mask=attention_mask,
                            position_ids=position_ids
                        ))
                        if args.aug_loss:
                            fp_inps_2[j] = extract_hidden_states(call_layer_forward(
                                qlayer,
                                quant_inps[j].unsqueeze(0),
                                layer_kwargs=layer_kwargs,
                                attention_mask=attention_mask,
                                position_ids=position_ids
                            ))
        # init smooth parameters
        set_quant_state(qlayer, weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
        qlayer.let = args.let
        use_shift = True 
        if is_llama or args.abits == 16:
            use_shift = False                   # deactivate channel-wise shifting for llama model and weight-only quantization
        if args.let:
            # init channel-wise scaling and shift
            qlayer.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.ones(layer.self_attn.q_proj.out_features,device=dev, dtype=dtype)))
            for name,module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    for key in pairs.keys():
                        if key in name:
                            act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype).clamp(min=1e-5)
                            weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                            scale = (act.pow(args.alpha)/weight.pow(1-args.alpha)).clamp(min=1e-5)
                            if use_shift and not is_llama:
                                shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype)
                            else:
                                shift = torch.zeros_like(scale)
                            qlayer.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift))
                            qlayer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale))
                                
        if args.resume:
            qlayer.load_state_dict(omni_parameters[i], strict=False)
        

        if args.epochs > 0:
            if use_decoupled_moe_training:
                final_loss = train_decoupled_moe_layer(
                    layer,
                    qlayer,
                    args,
                    logger,
                    i,
                    current_fp_layer_inputs,
                    quant_inps,
                    layer_kwargs,
                    attention_mask_batch,
                    attention_mask,
                    position_ids,
                    traincast,
                    quant_routing_top_n,
                    use_router_weight_in_loss,
                )
            else:
                with torch.no_grad():
                    qlayer.float()      # required for AMP training
                # create optimizer with parameter groups
                # LET/LWC parameters use weight_decay=0 (fixed, not controlled by args.wd)
                param_groups = [
                    {"params": let_parameters(qlayer, use_shift), "lr": args.let_lr, "weight_decay": 0},
                    {"params": lwc_parameters(qlayer), "lr": args.lwc_lr, "weight_decay": 0}
                ]
                
                # Add shared_expert_gate parameters if training is enabled (uses args.wd)
                if train_shared_gate:
                    shared_gate_params = []
                    for name, module in qlayer.named_modules():
                        if name.endswith("shared_expert_gate") and isinstance(module, nn.Linear):
                            shared_gate_params.extend([p for p in module.parameters() if p.requires_grad])
                    if shared_gate_params:
                        param_groups.append({"params": shared_gate_params, "lr": shared_gate_lr, "weight_decay": args.wd})
                
                # Add LoRA parameters if training is enabled (uses args.wd)
                if train_gate_lora:
                    lora_params = []
                    for name, module in qlayer.named_modules():
                        if isinstance(module, LoraLinear):
                            lora_params.extend([module.lora_A, module.lora_B])
                    if lora_params:
                        param_groups.append({"params": lora_params, "lr": gate_lora_lr, "weight_decay": args.wd})

                if getattr(args, 'use_linear_lora', False):
                    linear_lora_params = []
                    for name, module in qlayer.named_modules():
                        if isinstance(module, QuantLinear):
                            linear_lora_params.extend(module.get_lora_parameters())
                    if linear_lora_params:
                        param_groups.append({"params": linear_lora_params, "lr": args.linear_lora_lr, "weight_decay": args.wd})
                
                # Default weight_decay=0 for optimizer (each group specifies its own)
                optimizer = torch.optim.AdamW(param_groups, weight_decay=0)
                loss_scaler = utils.NativeScalerWithGradNormCount()
                
                # Collect all trainable parameters for gradient clipping
                # Start with quantization parameters (LET/LWC)
                clip_parameters = list(get_omni_parameters(qlayer, use_shift))
                
                # Add shared_expert_gate parameters if training is enabled
                if train_shared_gate:
                    for name, module in qlayer.named_modules():
                        if name.endswith("shared_expert_gate") and isinstance(module, nn.Linear):
                            clip_parameters.extend([p for p in module.parameters() if p.requires_grad])
                
                # Add LoRA parameters if training is enabled
                if train_gate_lora:
                    for name, module in qlayer.named_modules():
                        if isinstance(module, LoraLinear):
                            clip_parameters.extend([module.lora_A, module.lora_B])

                if getattr(args, 'use_linear_lora', False):
                    for name, module in qlayer.named_modules():
                        if isinstance(module, QuantLinear):
                            clip_parameters.extend(module.get_lora_parameters())
                
                # Log training configuration once per block (first layer only)
                if i == 0:
                    if train_shared_gate:
                        logger.info(f"[Gate Training] shared_expert_gate training ENABLED with lr={shared_gate_lr}")
                    if train_gate_lora:
                        logger.info(f"[Gate Training] router gate LoRA training ENABLED with r={lora_r}, alpha={lora_alpha}, lr={gate_lora_lr}")
                    if getattr(args, 'use_linear_lora', False):
                        logger.info(f"[Linear LoRA] QuantLinear LoRA ENABLED with r={args.linear_lora_r}, alpha={args.linear_lora_alpha}, lr={args.linear_lora_lr}")
                    if not train_shared_gate and not train_gate_lora and not getattr(args, 'use_linear_lora', False):
                        logger.info("[Gate Training] All gate training DISABLED (default behavior)")
                
                for epochs in range(args.epochs):
                    loss_list = []
                    norm_list = []
                    for j in range(args.nsamples//args.batch_size):    
                        index = j * args.batch_size
                        # obtain output of quantization model
                        with traincast():
                            smooth_and_quant_temporary(qlayer, args, is_llama)
                            quant_out = extract_hidden_states(call_layer_forward(
                                qlayer,
                                quant_inps[index:index+args.batch_size,],
                                layer_kwargs=layer_kwargs,
                                attention_mask=attention_mask_batch,
                                position_ids=position_ids
                            ))
                            loss = loss_func(fp_inps[index:index+args.batch_size,], quant_out)
                            if args.aug_loss:
                                loss += loss_func(fp_inps_2[index:index+args.batch_size,], quant_out)
                        if not math.isfinite(loss.item()):
                            logger.info("Loss is NAN, stopping training")
                            pdb.set_trace()
                            
                        loss_list.append(loss.detach().cpu())
                        optimizer.zero_grad()
                        # Use complete parameter list for gradient clipping
                        norm = loss_scaler(loss, optimizer, parameters=clip_parameters).cpu()
                        norm_list.append(norm.data)

                    loss_mean = torch.stack(loss_list).mean()
                    norm_mean = torch.stack(norm_list).mean()
                    
                    # Calculate and log gate training gradient norms
                    gate_grad_info = ""
                    shared_gate_grad_norm = 0.0
                    lora_grad_norm = 0.0
                    linear_lora_grad_norm = 0.0
                    
                    if train_shared_gate:
                        for name, module in qlayer.named_modules():
                            if name.endswith("shared_expert_gate") and isinstance(module, nn.Linear):
                                if module.weight.grad is not None:
                                    shared_gate_grad_norm += module.weight.grad.norm().item() ** 2
                        shared_gate_grad_norm = shared_gate_grad_norm ** 0.5
                        gate_grad_info += f" shared_gate_grad:{shared_gate_grad_norm:.2e}"
                    
                    if train_gate_lora:
                        for name, module in qlayer.named_modules():
                            if isinstance(module, LoraLinear):
                                if module.lora_A.grad is not None:
                                    lora_grad_norm += module.lora_A.grad.norm().item() ** 2
                                if module.lora_B.grad is not None:
                                    lora_grad_norm += module.lora_B.grad.norm().item() ** 2
                        lora_grad_norm = lora_grad_norm ** 0.5
                        gate_grad_info += f" lora_grad:{lora_grad_norm:.2e}"

                    if getattr(args, 'use_linear_lora', False):
                        for name, module in qlayer.named_modules():
                            if isinstance(module, QuantLinear):
                                for param in module.get_lora_parameters():
                                    if param.grad is not None:
                                        linear_lora_grad_norm += param.grad.norm().item() ** 2
                        linear_lora_grad_norm = linear_lora_grad_norm ** 0.5
                        gate_grad_info += f" linear_lora_grad:{linear_lora_grad_norm:.2e}"
                    
                    logger.info(f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean}{gate_grad_info} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
                    final_loss = loss_mean.item()  # always update; after loop ends this holds last layer's last epoch loss
                    
                    # WandB logging: Log metrics with strict naming schema for Router Collapse detection
                    if wandb is not None:
                        # Extract learning rates from optimizer param_groups
                        lr_shared_gate = None
                        lr_router_lora = None
                        lr_linear_lora = None
                        for pg in optimizer.param_groups:
                            # Identify groups by checking if they contain shared_gate or lora params
                            if len(pg['params']) > 0:
                                # Check if this is the shared gate group (3rd group, index 2)
                                if train_shared_gate and lr_shared_gate is None:
                                    for name, module in qlayer.named_modules():
                                        if name.endswith("shared_expert_gate") and isinstance(module, nn.Linear):
                                            for p in module.parameters():
                                                if p.requires_grad and any(p is pp for pp in pg['params']):
                                                    lr_shared_gate = pg['lr']
                                                    break
                                # Check if this is the lora group (4th group, index 3)
                                if train_gate_lora and lr_router_lora is None:
                                    for name, module in qlayer.named_modules():
                                        if isinstance(module, LoraLinear):
                                            if any(module.lora_A is pp or module.lora_B is pp for pp in pg['params']):
                                                lr_router_lora = pg['lr']
                                                break
                                if getattr(args, 'use_linear_lora', False) and lr_linear_lora is None:
                                    for name, module in qlayer.named_modules():
                                        if isinstance(module, QuantLinear):
                                            if any(param is pp for param in module.get_lora_parameters() for pp in pg['params']):
                                                lr_linear_lora = pg['lr']
                                                break
                        
                        # Build metrics dict with strict naming schema
                        wandb_metrics = {
                            "train/loss": loss_mean.item(),
                            "train/layer_id": i,
                            "train/epoch": epochs,
                            "train/grad_norm_mean": norm_mean.item(),
                        }
                        
                        # Add gradient norms for Router Collapse detection
                        if train_shared_gate:
                            wandb_metrics["grad/shared_expert_norm"] = shared_gate_grad_norm
                        if train_gate_lora:
                            wandb_metrics["grad/router_lora_norm"] = lora_grad_norm
                        if getattr(args, 'use_linear_lora', False):
                            wandb_metrics["grad/linear_lora_norm"] = linear_lora_grad_norm
                        
                        # Add learning rates for hyperparameter tracking
                        if lr_shared_gate is not None:
                            wandb_metrics["lr/shared_gate"] = lr_shared_gate
                        if lr_router_lora is not None:
                            wandb_metrics["lr/router_lora"] = lr_router_lora
                        if lr_linear_lora is not None:
                            wandb_metrics["lr/linear_lora"] = lr_linear_lora
                        if calibrate_router:
                            wandb_metrics["lr/router_lr"] = router_lr
                        
                        wandb.log(wandb_metrics, step=global_step)
                        global_step += 1
                clear_temp_variable(qlayer)
                del optimizer
                
                # Merge LoRA weights back into original Linear layers after training
                if train_gate_lora:
                    for name, module in list(qlayer.named_modules()):
                        if isinstance(module, LoraLinear):
                            merged_linear = module.merge()
                            add_new_module(name, qlayer, merged_linear)
                            logger.info(f"Merged LoRA weights for {name}")

        if args.epochs > 0 and train_gate_lora and use_decoupled_moe_training:
            for name, module in list(qlayer.named_modules()):
                if isinstance(module, LoraLinear):
                    merged_linear = module.merge()
                    add_new_module(name, qlayer, merged_linear)
                    logger.info(f"Merged LoRA weights for {name}")

        if args.epochs > 0 and use_decoupled_moe_training:
            set_quant_state(qlayer, weight_quant=False, act_quant=False)
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    for j in range(args.nsamples):
                        fp_inps[j] = extract_hidden_states(call_layer_forward(
                            layer,
                            current_fp_layer_inputs[j].unsqueeze(0),
                            layer_kwargs=layer_kwargs,
                            attention_mask=attention_mask,
                            position_ids=position_ids
                        ))
                        if args.aug_loss:
                            fp_inps_2[j] = extract_hidden_states(call_layer_forward(
                                qlayer,
                                quant_inps[j].unsqueeze(0),
                                layer_kwargs=layer_kwargs,
                                attention_mask=attention_mask,
                                position_ids=position_ids
                            ))
        
        # =================================================================
        # Post-LWC Expert Shift Check (always for Qwen MoE)
        # =================================================================
        if is_qwen_moe and cached_router_labels is not None:
            teacher_logits, teacher_indices = cached_router_labels
            seqlen = fp_inps.shape[1]
            
            # Create temp_weight ONCE for all samples
            smooth_and_quant_temporary(qlayer, args, is_llama)
            
            with torch.no_grad():
                post_lwc_shift_any_sum = 0.0
                post_lwc_shift_half_sum = 0.0
                post_lwc_shift_all_sum = 0.0
                num_samples = min(args.nsamples, 8)
                for j in range(num_samples):
                    # Use hook-based forward to get router logits
                    with torch.amp.autocast('cuda'):
                        out, router_logits = forward_with_router_logits(
                            qlayer,
                            quant_inps[j].unsqueeze(0),
                            layer_kwargs=layer_kwargs,
                            attention_mask=attention_mask,
                            position_ids=position_ids
                        )
                    
                    if router_logits is not None:
                        # Reshape router_logits if needed
                        if router_logits.dim() == 2:
                            router_logits = router_logits.unsqueeze(0)
                        
                        # Get correct teacher indices for this sample
                        if teacher_indices.dim() == 2:
                            teacher_idx_sample = teacher_indices[j*seqlen:(j+1)*seqlen].unsqueeze(0)
                        else:
                            teacher_idx_sample = teacher_indices[j:j+1]
                        
                        shift_metrics = compute_expert_shift_detailed(
                            router_logits,
                            teacher_idx_sample,
                            k_routing
                        )
                        post_lwc_shift_any_sum += shift_metrics["shift_any"]
                        post_lwc_shift_half_sum += shift_metrics["shift_half"]
                        post_lwc_shift_all_sum += shift_metrics["shift_all"]
                
                post_lwc_shift_any = post_lwc_shift_any_sum / num_samples
                post_lwc_shift_half = post_lwc_shift_half_sum / num_samples
                post_lwc_shift_all = post_lwc_shift_all_sum / num_samples
                logger.info(f"[Expert Shift] Layer {i}: Post-LWC Expert Shift - Any: {post_lwc_shift_any:.4f}, Half: {post_lwc_shift_half:.4f}, All: {post_lwc_shift_all:.4f}")
                
                # Log improvement from Pre-LWC to Post-LWC
                if pre_lwc_shift is not None:
                    lwc_improvement_any = pre_lwc_shift[0] - post_lwc_shift_any
                    lwc_improvement_half = pre_lwc_shift[1] - post_lwc_shift_half
                    lwc_improvement_all = pre_lwc_shift[2] - post_lwc_shift_all
                    logger.info(f"[Expert Shift] Layer {i}: LWC Improvement (Pre-LWC -> Post-LWC) - Any: {lwc_improvement_any:.4f}, Half: {lwc_improvement_half:.4f}, All: {lwc_improvement_all:.4f}")
                    
                    # Collect data for visualization (3 phases)
                    expert_shift_data.append({
                        "layer": i,
                        "initial_any": pre_lwc_shift[0],
                        "initial_half": pre_lwc_shift[1],
                        "initial_all": pre_lwc_shift[2],
                        "post_calib_any": post_calib_shift[0] if post_calib_shift else None,
                        "post_calib_half": post_calib_shift[1] if post_calib_shift else None,
                        "post_calib_all": post_calib_shift[2] if post_calib_shift else None,
                        "post_lwc_any": post_lwc_shift_any,
                        "post_lwc_half": post_lwc_shift_half,
                        "post_lwc_all": post_lwc_shift_all,
                    })
            
            # Clean up temp_weight
            clear_temp_variable(qlayer)
        
        qlayer.half() 
        # real smooth and quantization
        smooth_and_quant_inplace(qlayer, args, is_llama)
        if args.epochs>0:
            # update input of quantization model
            with torch.no_grad():
                # with torch.cuda.amp.autocast():
                with traincast():
                    for j in range(args.nsamples):
                        quant_inps[j] = extract_hidden_states(call_layer_forward(
                            qlayer,
                            quant_inps[j].unsqueeze(0),
                            layer_kwargs=layer_kwargs,
                            attention_mask=attention_mask,
                            position_ids=position_ids
                        ))
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
            omni_parameters[i] = omni_state_dict(qlayer)
            torch.save(omni_parameters, os.path.join(args.output_dir, f"omni_parameters.pth"))
        else:
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
        if args.real_quant:
            assert args.wbits in [2,3,4] and args.abits >= 16   # only support weight-only quantization
            named_linears = get_named_linears(qlayer)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scales
                zeros = module.weight_quantizer.zeros
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1)
                zeros = zeros.view(dim0,-1)
                if args.wbits == 3:
                    q_linear = qlinear_cuda.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                else:
                    q_linear = qlinear_triton.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                add_new_module(name, qlayer, q_linear)       
                print(f"pack quantized {name} finished")
                del module        
        del layer
        torch.cuda.empty_cache()

    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    
    # === WandB Expert Shift Visualization (Matplotlib, 3-Phase Line Charts) ===
    # 3 separate charts: shift_any, shift_half, shift_all
    # Each chart: X=layer_id, lines=Initial / Post-Calib(if exists) / Post-LWC
    if wandb is not None and len(expert_shift_data) > 0:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            has_calib = any(e.get("post_calib_any") is not None for e in expert_shift_data)
            layers = [e["layer"] for e in expert_shift_data]

            phase_defs = [
                ("initial", "Initial", {"color": "#1f77b4", "linestyle": "-", "marker": "o"}),
                ("post_calib", "Post-Calib", {"color": "#ff7f0e", "linestyle": "--", "marker": "s"}),
                ("post_lwc", "Post-LWC", {"color": "#2ca02c", "linestyle": "-.", "marker": "^"}),
            ]

            fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
            for metric, ax in zip(["any", "half", "all"], axes):
                for phase_key, phase_label, style in phase_defs:
                    if phase_key == "post_calib" and not has_calib:
                        continue
                    xs = []
                    ys = []
                    for entry in expert_shift_data:
                        value = entry.get(f"{phase_key}_{metric}")
                        if value is None:
                            continue
                        xs.append(entry["layer"])
                        ys.append(value)
                    if xs:
                        ax.plot(xs, ys, label=phase_label, **style)
                ax.set_title(f"shift_{metric}")
                ax.set_xlabel("Layer")
                ax.set_ylabel("Shift")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="best")

            wandb.log({
                "expert_shift/three_phase": wandb.Image(
                    fig, caption="Expert Shift: Initial vs Post-Calib vs Post-LWC"
                )
            })
            plt.close(fig)
            logger.info(f"[Expert Shift] Uploaded Matplotlib visualization to WandB ({len(expert_shift_data)} layers)")
        except ImportError:
            logger.warning("[Expert Shift] Matplotlib not installed; skipping custom visualization.")
        except Exception as e:
            logger.warning(f"[Expert Shift] Failed to create Matplotlib visualization: {e}")
    
    return model, final_loss

