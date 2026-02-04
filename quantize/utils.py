from collections import OrderedDict
from quantize.int_linear import QuantLinear
import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.int_matmul import QuantMatMul
from models.transformation import *


# =============================================================================
# Task 1: Router Label Caching (Pre-computation) for Qwen2-MoE
# =============================================================================
@torch.no_grad()
def capture_router_labels(model, dataloader, dev, topk=20, seqlen=2048, nsamples=128, logger=None):
    """
    Capture FP16 router labels (top-k indices and probabilities) for all layers.
    
    Args:
        model: The full model (e.g., Qwen2MoeForCausalLM)
        dataloader: Calibration dataloader
        dev: Target device (GPU)
        topk: Number of top experts to cache
        seqlen: Sequence length
        nsamples: Number of samples to process
        logger: Optional logger for info messages
    
    Returns:
        cached_labels: {layer_idx: (values_tensor, indices_tensor)}
                      values_tensor: [nsamples, seqlen, topk] - probability values
                      indices_tensor: [nsamples, seqlen, topk] - expert indices
    """
    layers = model.model.layers
    num_layers = len(layers)
    
    # Storage for captured logits per layer
    cached_logits = {i: [] for i in range(num_layers)}
    hooks = []
    
    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output is router logits: [batch, seq_len, num_experts]
            logits = output.detach()
            probs = torch.softmax(logits, dim=-1)
            values, indices = torch.topk(probs, k=topk, dim=-1)
            # Store on GPU (do not move to CPU)
            cached_logits[layer_idx].append((values, indices))
        return hook_fn
    
    # Register hooks on mlp.gate for all layers
    for i, layer in enumerate(layers):
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
            hook = layer.mlp.gate.register_forward_hook(make_hook(i))
            hooks.append(hook)
    
    # Move model to device for inference
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    
    # Run inference on calibration data
    sample_count = 0
    for batch in dataloader:
        if sample_count >= nsamples:
            break
        try:
            # Move batch to device
            input_ids = batch[0].to(dev)
            # Move required layers to device temporarily
            for layer in layers:
                layer.to(dev)
            model(input_ids)
            # Move layers back to CPU to save memory
            for layer in layers:
                layer.to('cpu')
            sample_count += input_ids.shape[0]
        except Exception as e:
            if logger:
                logger.warning(f"Error during router label capture: {e}")
            continue
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Move embeddings back to CPU
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    
    # Concatenate captured labels for each layer
    cached_labels = {}
    for layer_idx in range(num_layers):
        if cached_logits[layer_idx]:
            all_values = torch.cat([v for v, _ in cached_logits[layer_idx]], dim=0)
            all_indices = torch.cat([i for _, i in cached_logits[layer_idx]], dim=0)
            cached_labels[layer_idx] = (all_values, all_indices)
            if logger:
                logger.info(f"Layer {layer_idx}: captured router labels shape: values={all_values.shape}, indices={all_indices.shape}")
    
    return cached_labels


@torch.no_grad()
def capture_router_labels_layerwise(layer, inps, attention_mask, position_ids, dev, topk=20, logger=None):
    """
    Capture FP16 router labels for a single layer using pre-captured inputs.
    This is more memory-efficient than processing the entire model at once.
    
    Args:
        layer: A single decoder layer (on device)
        inps: Input hidden states [nsamples, seqlen, hidden_size]
        attention_mask: Attention mask tensor
        position_ids: Position IDs tensor
        dev: Target device
        topk: Number of top experts to cache
        logger: Optional logger
    
    Returns:
        (values_tensor, indices_tensor) on GPU device
    """
    if not hasattr(layer, 'mlp') or not hasattr(layer.mlp, 'gate'):
        return None
    
    all_values = []
    all_indices = []
    captured_data = []
    
    def hook_fn(module, input, output):
        logits = output.detach()
        probs = torch.softmax(logits, dim=-1)
        values, indices = torch.topk(probs, k=topk, dim=-1)
        captured_data.append((values, indices))
    
    hook = layer.mlp.gate.register_forward_hook(hook_fn)
    
    nsamples = inps.shape[0]
    for j in range(nsamples):
        captured_data.clear()
        with torch.cuda.amp.autocast():
            _ = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)
        if captured_data:
            all_values.append(captured_data[0][0])
            all_indices.append(captured_data[0][1])
    
    hook.remove()
    
    if all_values:
        values_tensor = torch.cat(all_values, dim=0)  # [nsamples, seqlen, topk]
        indices_tensor = torch.cat(all_indices, dim=0)
        if logger:
            logger.info(f"Captured router labels: values={values_tensor.shape}, indices={indices_tensor.shape}")
        return (values_tensor, indices_tensor)
    
    return None


# =============================================================================
# Task 2: Expert Shift Metrics
# =============================================================================
def compute_expert_shift(student_logits, teacher_indices, k_routing):
    """
    Compute the mismatch rate between student and teacher expert selections.
    
    Args:
        student_logits: Router logits from student model [batch, seq_len, num_experts]
        teacher_indices: Top-k expert indices from teacher [batch, seq_len, topk_cached]
        k_routing: Number of experts actually used for routing (may be <= topk_cached)
    
    Returns:
        mismatch_rate: Float value representing percentage of tokens where at least one expert changed
    """
    # Get student's top-k selections
    student_probs = torch.softmax(student_logits, dim=-1)
    _, student_indices = torch.topk(student_probs, k=k_routing, dim=-1)  # [batch, seq, k_routing]
    
    # Take only top k_routing from teacher indices
    teacher_topk = teacher_indices[..., :k_routing]  # [batch, seq, k_routing]
    
    # Sort both for set comparison
    student_sorted, _ = torch.sort(student_indices, dim=-1)
    teacher_sorted, _ = torch.sort(teacher_topk, dim=-1)
    
    # Compare: a token is "matched" if all selected experts are the same
    match = (student_sorted == teacher_sorted).all(dim=-1)  # [batch, seq]
    
    # Compute mismatch rate (at least one expert changed)
    mismatch_rate = 1.0 - match.float().mean().item()
    
    return mismatch_rate


def compute_expert_shift_detailed(student_logits, teacher_indices, k_routing):
    """
    Compute detailed expert shift metrics based on three levels of change.
    
    Args:
        student_logits: Router logits from student model [batch, seq_len, num_experts]
        teacher_indices: Top-k expert indices from teacher [batch, seq_len, topk_cached]
        k_routing: Number of experts actually used for routing (may be <= topk_cached)
    
    Returns:
        dict with:
            - shift_any: Rate of tokens where at least one expert changed
            - shift_half: Rate of tokens where at least half of experts changed
            - shift_all: Rate of tokens where all experts changed
    """
    student_probs = torch.softmax(student_logits, dim=-1)
    _, student_indices = torch.topk(student_probs, k=k_routing, dim=-1)  # [batch, seq, k_routing]
    
    teacher_topk = teacher_indices[..., :k_routing]  # [batch, seq, k_routing]
    
    # Sort both for set comparison
    student_sorted, _ = torch.sort(student_indices, dim=-1)
    teacher_sorted, _ = torch.sort(teacher_topk, dim=-1)
    
    # Count number of matching experts per token
    # After sorting, we can compare element-wise
    matches_per_token = (student_sorted == teacher_sorted).sum(dim=-1)  # [batch, seq]
    
    # Calculate the three shift metrics
    total_tokens = matches_per_token.numel()
    
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
# Task 3: Quantized Qwen2-MoE Decoder Layer Wrapper
# =============================================================================
class QuantQwen2MoeDecoderLayer(nn.Module):
    """
    Wrapper for Qwen2MoeDecoderLayer that supports outputting router logits.
    This preserves backward compatibility while enabling router calibration.
    """
    
    def __init__(self, original_layer, args=None):
        super().__init__()
        self.layer = original_layer
        self.args = args
        
        # Store reference to the router gate for easy access
        self._router_logits = None
        self._hook_handle = None
        
        # Track if we need to capture router logits
        self._capture_router_logits = False
    
    def _setup_router_hook(self):
        """Setup hook to capture router logits during forward pass."""
        if self._hook_handle is not None:
            return
        
        if hasattr(self.layer, 'mlp') and hasattr(self.layer.mlp, 'gate'):
            def hook_fn(module, input, output):
                # Keep gradient history by not detaching
                self._router_logits = output
            self._hook_handle = self.layer.mlp.gate.register_forward_hook(hook_fn)
    
    def _remove_router_hook(self):
        """Remove the router hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
    
    def forward(self, hidden_states, attention_mask=None, position_ids=None, 
                output_router_logits=False, **kwargs):
        """
        Forward pass with optional router logits output.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Attention mask
            position_ids: Position IDs
            output_router_logits: If True, return (hidden_states, router_logits)
            **kwargs: Additional arguments passed to the layer
        
        Returns:
            If output_router_logits=False: hidden_states (or tuple from original layer)
            If output_router_logits=True: (hidden_states, router_logits)
        """
        self._router_logits = None
        
        if output_router_logits:
            self._setup_router_hook()
        
        # Call the original layer
        outputs = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs
        )
        
        # Extract hidden states from outputs
        if isinstance(outputs, tuple):
            hidden_states_out = outputs[0]
        else:
            hidden_states_out = outputs
        
        if output_router_logits:
            self._remove_router_hook()
            router_logits = self._router_logits
            self._router_logits = None
            return (hidden_states_out, router_logits)
        
        return outputs
    
    def __getattr__(self, name):
        """Delegate attribute access to the wrapped layer."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.layer, name)
    
    def named_modules(self, memo=None, prefix='', remove_duplicate=True):
        """Include wrapped layer's modules in named_modules iteration."""
        yield from super().named_modules(memo, prefix, remove_duplicate)
        yield from self.layer.named_modules(memo, prefix + 'layer.' if prefix else 'layer.', remove_duplicate)
    
    def named_parameters(self, prefix='', recurse=True):
        """Include wrapped layer's parameters."""
        yield from self.layer.named_parameters(prefix=prefix, recurse=recurse)


def wrap_qwen2moe_layer_for_router_output(qlayer):
    """
    Wrap a Qwen2MoE layer to support router logits output.
    This is a non-invasive wrapper that preserves the original layer structure.
    
    Args:
        qlayer: The quantized layer (with QuantLinear modules)
    
    Returns:
        Wrapped layer that supports output_router_logits argument
    """
    # Check if already wrapped
    if isinstance(qlayer, QuantQwen2MoeDecoderLayer):
        return qlayer
    
    return QuantQwen2MoeDecoderLayer(qlayer)


# =============================================================================
# Router Calibration Loss Functions
# =============================================================================
def compute_topk_mse_loss(student_logits, teacher_probs, teacher_indices):
    """
    Compute TopK-MSE loss for router calibration.
    
    Args:
        student_logits: Router logits from student [batch, seq, num_experts]
        teacher_probs: Top-k probability values from teacher [batch, seq, topk]
        teacher_indices: Top-k expert indices from teacher [batch, seq, topk]
    
    Returns:
        loss: MSE loss between gathered student probs and teacher probs
    """
    # Compute student probabilities
    student_probs = torch.softmax(student_logits, dim=-1)  # [batch, seq, num_experts]
    
    # Gather student probabilities at teacher's top-k indices
    # teacher_indices: [batch, seq, topk]
    gathered_student_probs = torch.gather(student_probs, dim=-1, index=teacher_indices)
    
    # Compute MSE loss
    loss = F.mse_loss(gathered_student_probs, teacher_probs)
    
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

class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = truncated_tensor[truncated_tensor.abs() < threshold].sign() * threshold
        return truncated_tensor
        

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

     
def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)     

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
                module.temp_weight = module.weight
    # quant
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            if hasattr(module, "temp_weight"):
                module.temp_weight = module.weight_quantizer(module.temp_weight)
            else:
                module.temp_weight = module.weight_quantizer(module.weight)
            if not hasattr(module, "temp_bias"):
                module.temp_bias = module.bias
            module.use_temporary_parameter=True
            
def clear_temp_variable(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
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
            module.weight = module.weight_quantizer(module.weight)
            module.use_temporary_parameter=False

def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
    # setting weight quantization here does not affect actual forward pass
    self.use_weight_quant = weight_quant
    self.use_act_quant = act_quant
    for m in self.modules():
        if isinstance(m, (QuantLinear, QuantMatMul)):
            m.set_quant_state(weight_quant, act_quant)
