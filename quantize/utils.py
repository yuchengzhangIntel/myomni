import inspect
from collections import OrderedDict
from quantize.int_linear import QuantLinear
import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.int_matmul import QuantMatMul
from models.transformation import *


def get_cuda_amp_dtype():
    if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def get_layer_forward_kwargs(layer, layer_kwargs=None, **extra_kwargs):
    merged_kwargs = {}
    if layer_kwargs:
        merged_kwargs.update(layer_kwargs)
    for key, value in extra_kwargs.items():
        if value is not None:
            merged_kwargs[key] = value

    signature = inspect.signature(layer.forward)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return merged_kwargs

    return {
        key: value
        for key, value in merged_kwargs.items()
        if value is not None and key in signature.parameters
    }


def call_layer_forward(layer, hidden_states, layer_kwargs=None, **extra_kwargs):
    return layer(
        hidden_states,
        **get_layer_forward_kwargs(layer, layer_kwargs=layer_kwargs, **extra_kwargs)
    )


def extract_hidden_states(outputs):
    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        return outputs.hidden_states
    return outputs[0]


@torch.no_grad()
def capture_router_labels_layerwise(layer, inps, dev, topk=20, logger=None, layer_kwargs=None):
    """
    Capture mixed-precision router labels for a single layer using pre-captured inputs.
    This is more memory-efficient than processing the entire model at once.
    
    Args:
        layer: A single decoder layer (on device)
        inps: Input hidden states [nsamples, seqlen, hidden_size]
        dev: Target device
        topk: Number of top experts to cache
        logger: Optional logger
        layer_kwargs: Optional keyword arguments forwarded to layer.forward
    
    Returns:
        (values_tensor, indices_tensor) on GPU device
        Shape: [nsamples, seqlen, topk]
    """
    if not hasattr(layer, 'mlp') or not hasattr(layer.mlp, 'gate'):
        return None
    
    nsamples = inps.shape[0]
    seqlen = inps.shape[1]
    all_values = []
    all_indices = []
    captured_data = []
    
    def hook_fn(module, input, output):
        # Router gate output shape can be [batch*seq_len, num_experts] or [batch, seq_len, num_experts]
        logits = output.detach()
        # Use raw logits (no softmax) for topk selection and loss computation
        logits_f32 = logits.float()
        values, indices = torch.topk(logits_f32, k=topk, dim=-1)
        # Keep values in float32 to avoid NaN issues later
        captured_data.append((values, indices))
    
    hook = layer.mlp.gate.register_forward_hook(hook_fn)
    
    for j in range(nsamples):
        captured_data.clear()
        with torch.autocast(device_type='cuda', dtype=get_cuda_amp_dtype()):
            _ = call_layer_forward(layer, inps[j].unsqueeze(0), layer_kwargs=layer_kwargs)
        if captured_data:
            values, indices = captured_data[0]
            # Ensure shape is [1, seqlen, topk] - reshape if needed
            if values.dim() == 2:
                # Shape is [seq_len, topk], add batch dimension
                values = values.unsqueeze(0)  # [1, seq_len, topk]
                indices = indices.unsqueeze(0)
            all_values.append(values)
            all_indices.append(indices)
    
    hook.remove()
    
    if all_values:
        values_tensor = torch.cat(all_values, dim=0)  # [nsamples, seqlen, topk]
        indices_tensor = torch.cat(all_indices, dim=0)
        if logger:
            logger.info(f"Captured router labels: values={values_tensor.shape}, indices={indices_tensor.shape}")
            logger.info(f"[DEBUG] Expected shape: [{nsamples}, {seqlen}, {topk}]")
            if values_tensor.shape != (nsamples, seqlen, topk):
                logger.warning(f"[WARNING] Shape mismatch! Got {values_tensor.shape}, expected ({nsamples}, {seqlen}, {topk})")
        return (values_tensor, indices_tensor)
    
    return None


# =============================================================================
# Task 2: Expert Shift Metrics
# =============================================================================
def compute_expert_shift_detailed(student_logits, teacher_indices, k_routing, debug=False):
    """
    Compute detailed expert shift metrics based on three levels of change.
    Uses proper SET intersection (one-hot encoding) instead of sorted comparison.
    
    Args:
        student_logits: Router logits from student model [batch, seq_len, num_experts]
        teacher_indices: Top-k expert indices from teacher [batch, seq_len, topk_cached]
        k_routing: Number of experts actually used for routing (may be <= topk_cached)
        debug: If True, print debug information
    
    Returns:
        dict with:
            - shift_any: Rate of tokens where at least one expert changed
            - shift_half: Rate of tokens where at least half of experts changed
            - shift_all: Rate of tokens where all experts changed
    """
    num_experts = student_logits.shape[-1]
    batch_size, seq_len = student_logits.shape[:2]
    
    student_probs = torch.softmax(student_logits, dim=-1)
    _, student_indices = torch.topk(student_probs, k=k_routing, dim=-1)  # [batch, seq, k_routing]
    
    teacher_topk = teacher_indices[..., :k_routing]  # [batch, seq, k_routing]
    
    # Use one-hot encoding for proper SET intersection
    # This correctly handles cases like {0,1,2,3} vs {1,2,3,4} -> 3 matches
    student_onehot = torch.zeros(batch_size, seq_len, num_experts, device=student_logits.device)
    student_onehot.scatter_(-1, student_indices, 1)
    
    teacher_onehot = torch.zeros(batch_size, seq_len, num_experts, device=student_logits.device)
    teacher_onehot.scatter_(-1, teacher_topk.to(student_logits.device), 1)
    
    # Count intersection: experts selected by BOTH student and teacher
    matches_per_token = (student_onehot * teacher_onehot).sum(dim=-1)  # [batch, seq]
    
    if debug:
        # Print debug info for first few tokens
        print(f"[DEBUG] compute_expert_shift_detailed:")
        print(f"  student_logits shape: {student_logits.shape}, dtype: {student_logits.dtype}")
        print(f"  teacher_indices shape: {teacher_indices.shape}, dtype: {teacher_indices.dtype}")
        print(f"  k_routing: {k_routing}, num_experts: {num_experts}")
        print(f"  First token - Student top-{k_routing}: {student_indices[0, 0].tolist()}")
        print(f"  First token - Teacher top-{k_routing}: {teacher_topk[0, 0].tolist()}")
        print(f"  First token - matches: {matches_per_token[0, 0].item()}")
        print(f"  matches_per_token stats: min={matches_per_token.min().item()}, max={matches_per_token.max().item()}, mean={matches_per_token.float().mean().item():.4f}")
    
    # Calculate the three shift metrics
    # 1. At least one expert changed (any mismatch)
    at_least_one_changed = (matches_per_token < k_routing)  # not all match
    shift_any = at_least_one_changed.float().mean().item()
    
    # 2. At least half of experts changed
    half_k = k_routing / 2.0
    at_least_half_changed = (matches_per_token <= k_routing - half_k)  # half or more changed
    shift_half = at_least_half_changed.float().mean().item()
    
    # 3. All experts changed (no match)
    all_changed = (matches_per_token == 0)
    shift_all = all_changed.float().mean().item()
    
    return {
        "shift_any": shift_any,      # At least one expert changed
        "shift_half": shift_half,    # At least half of experts changed  
        "shift_all": shift_all,      # All experts changed
    }


# =============================================================================
# Task 3: Router Logits Capture via Simple Hook
# =============================================================================
def forward_with_router_logits(layer, hidden_states, layer_kwargs=None, **kwargs):
    """
    Forward pass through a layer while capturing router logits via hook.
    
    Args:
        layer: The decoder layer (can be quantized or original)
        hidden_states: Input tensor
        layer_kwargs: Optional keyword arguments forwarded to layer.forward
        **kwargs: Additional arguments
    
    Returns:
        (hidden_states_out, router_logits): Tuple of output hidden states and router logits
    """
    captured = {}
    
    def hook_fn(module, input, output):
        captured['logits'] = output  # Keep gradient history by not detaching
    
    # Register hook
    handle = None
    if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
        handle = layer.mlp.gate.register_forward_hook(hook_fn)
    
    # Forward
    outputs = call_layer_forward(layer, hidden_states, layer_kwargs=layer_kwargs, **kwargs)
    
    # Remove hook
    if handle is not None:
        handle.remove()
    
    # Extract hidden states
    hidden_states_out = extract_hidden_states(outputs)
    
    return hidden_states_out, captured.get('logits', None)


def create_router_hook():
    """
    Create a simple router logits hook using closure.
    
    Returns:
        (hook_fn, get_logits_fn): A tuple of hook function and getter function
    
    Usage:
        hook_fn, get_logits = create_router_hook()
        handle = layer.mlp.gate.register_forward_hook(hook_fn)
        _ = layer(input, ...)
        router_logits = get_logits()
        handle.remove()
    """
    captured = {}
    
    def hook_fn(module, input, output):
        captured['logits'] = output
    
    def get_logits():
        return captured.get('logits', None)
    
    def clear():
        captured.clear()
    
    return hook_fn, get_logits, clear


# =============================================================================
# Router Calibration Loss Functions
# =============================================================================
def compute_topk_mse_loss(student_logits, teacher_logits, teacher_indices, debug=False):
    """
    Compute TopK-MSE loss for router calibration using raw logits (no softmax).
    
    Args:
        student_logits: Router logits from student [batch, seq, num_experts]
        teacher_logits: Top-k raw logit values from teacher [batch, seq, topk]
        teacher_indices: Top-k expert indices from teacher [batch, seq, topk]
        debug: If True, print debug information
    
    Returns:
        loss: MSE loss between gathered student logits and teacher logits at top-k positions
    """
    # Use float32 for numerical stability
    student_logits_f32 = student_logits.float()
    
    # Ensure teacher tensors are on same device and use float32
    teacher_indices = teacher_indices.to(student_logits.device)
    teacher_logits_f32 = teacher_logits.to(device=student_logits.device).float()
    
    # Gather student raw logits at teacher's top-k indices
    gathered_student_logits = torch.gather(student_logits_f32, dim=-1, index=teacher_indices)
    
    if debug:
        print(f"[DEBUG] compute_topk_mse_loss (raw logits):")
        print(f"  student_logits: min={student_logits.min().item():.4f}, max={student_logits.max().item():.4f}")
        print(f"  gathered_student_logits: min={gathered_student_logits.min().item():.6f}, max={gathered_student_logits.max().item():.6f}")
        print(f"  teacher_logits: min={teacher_logits_f32.min().item():.6f}, max={teacher_logits_f32.max().item():.6f}")
        print(f"  Any NaN in student_logits: {torch.isnan(student_logits_f32).any().item()}")
        print(f"  Any NaN in teacher_logits: {torch.isnan(teacher_logits_f32).any().item()}")
    
    # Check for NaN and handle gracefully
    if torch.isnan(student_logits_f32).any() or torch.isnan(teacher_logits_f32).any():
        print(f"[WARNING] NaN detected in compute_topk_mse_loss!")
        # Return zero loss to avoid corrupting gradients
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True)
    
    # Compute MSE loss on raw logits in float32
    loss = F.mse_loss(gathered_student_logits, teacher_logits_f32)
    
    return loss


def let_parameters(model, use_shift=True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for n, m in model.named_parameters():
        if n.find(template) > -1:
            params.append(m)
    return iter(params)  

def lwc_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1:
            params.append(m)
    return iter(params)  

def get_omni_parameters(model, use_shift=True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1 or n.find(template) > -1:
            params.append(m)
    return iter(params)  

def omni_state_dict(model, destination=None, prefix='', keep_vars=False):
    if destination is None:
        destination = OrderedDict()
    for name, param in model.named_parameters():
        if name.find('smooth') > -1 or name.find('bound_factor') > -1:
            destination[prefix + name] = param if keep_vars else param.detach()
    return destination

def register_scales_and_zeros(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight_quantizer.register_scales_and_zeros()


def smooth_and_quant_temporary(model, args, isllama):
    if args.let:
        with torch.no_grad():
            for name, module in model.named_parameters():
                if "smooth_scale" in name:
                    module.data = truncate_number(module)
        if isllama:
            smooth_ln_fcs_temporary(model.input_layernorm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_temporary(model.post_attention_layernorm,[model.mlp.up_proj,model.mlp.gate_proj],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_fc_fc_temporary(model.self_attn.v_proj,model.self_attn.o_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
            smooth_q_k_temporary(model.self_attn.q_proj, model.self_attn.k_proj,
                                model.qkt_smooth_scale)
            model.mlp.down_proj.temp_weight = model.mlp.down_proj.weight
        else:
            smooth_ln_fcs_temporary(model.self_attn_layer_norm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_temporary(model.final_layer_norm,[model.fc1],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_ln_fcs_temporary(model.self_attn.v_proj,model.self_attn.out_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
            smooth_q_k_temporary(model.self_attn.q_proj, model.self_attn.k_proj,
                                model.qkt_smooth_scale)
            model.fc2.temp_weight = model.fc2.weight
    else:
        for name, module in model.named_modules():
            if isinstance(module, QuantLinear):
                module.temp_weight = module.get_effective_weight()
    # quant
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            if hasattr(module, "temp_weight"):
                module.temp_weight = module.weight_quantizer(module.temp_weight)
            else:
                module.temp_weight = module.weight_quantizer(module.get_effective_weight())
            if not hasattr(module, "temp_bias"):
                module.temp_bias = module.bias
            module.use_temporary_parameter=True
            
def clear_temp_variable(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.use_temporary_parameter = False  # Must reset this flag!
            if hasattr(module, "temp_weight"):
                del module.temp_weight
            if hasattr(module, "temp_bias"):
                del module.temp_bias

@torch.no_grad()   
def smooth_and_quant_inplace(model, args, isllama):
    if args.let:
        for name, module in model.named_parameters():
            if "smooth_scale" in name:
                module.data = truncate_number(module)
        if isllama:
            smooth_ln_fcs_inplace(model.input_layernorm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_inplace(model.post_attention_layernorm,[model.mlp.up_proj,model.mlp.gate_proj],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_fc_fc_inplace(model.self_attn.v_proj,model.self_attn.o_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
        else: # opt
            smooth_ln_fcs_inplace(model.self_attn_layer_norm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_inplace(model.final_layer_norm,[model.fc1],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_fc_fc_inplace(model.self_attn.v_proj,model.self_attn.out_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
        smooth_q_k_inplace(model.self_attn.q_proj, model.self_attn.k_proj,
                            model.qkt_smooth_scale)
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.merge_lora()
            module.weight = module.weight_quantizer(module.weight)
            module.use_temporary_parameter=False

def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
    # setting weight quantization here does not affect actual forward pass
    self.use_weight_quant = weight_quant
    self.use_act_quant = act_quant
    for m in self.modules():
        if isinstance(m, (QuantLinear, QuantMatMul)):
            m.set_quant_state(weight_quant, act_quant)
