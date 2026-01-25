"""
LoRA utilities for loading and applying LoRA adapters to Mixtral router layers.
This module provides functions to:
1. Define LoraLinear class (reused from omniquant.py)
2. Detect and replace router layers with LoRA adapters
3. Load LoRA weights from checkpoints while keeping base model frozen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from typing import Dict, Optional, Tuple


class LoraLinear(nn.Module):
    """
    LoRA adapter wrapper for a linear layer.
    Freezes the original weights and adds trainable low-rank matrices lora_A and lora_B.
    Output = original_output + (x @ lora_A @ lora_B) * scaling
    """
    def __init__(self, org_module: nn.Linear, rank: int = 8, lora_alpha: float = 16.0):
        super().__init__()
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        
        # Keep original weights frozen (use register_buffer to ensure they're not updated)
        self.register_buffer('weight', org_module.weight.data.clone())
        if org_module.bias is not None:
            self.register_buffer('bias', org_module.bias.data.clone())
        else:
            self.bias = None
        
        # LoRA parameters (trainable)
        self.lora_A = nn.Parameter(torch.zeros(self.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_features))
        self.scaling = lora_alpha / rank
        
        # Initialize lora_A with kaiming uniform and lora_B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original frozen linear output
        base_out = F.linear(x, self.weight, self.bias)
        # LoRA path: x @ lora_A @ lora_B * scaling
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out
    
    def merge_lora_weights(self) -> nn.Linear:
        """
        Merge LoRA weights into the base linear layer for inference efficiency.
        Returns a standard nn.Linear with merged weights.
        """
        merged_weight = self.weight + (self.lora_A @ self.lora_B).T * self.scaling
        linear = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        linear.weight.data = merged_weight
        if self.bias is not None:
            linear.bias.data = self.bias.clone()
        return linear


def is_mixtral_model(model_name: str) -> bool:
    """Check if the model is a Mixtral model based on its name."""
    return 'mixtral' in model_name.lower()


def get_router_layers(model) -> Dict[str, nn.Linear]:
    """
    Find all router (gate) layers in a Mixtral model.
    Router layers are used to route tokens to different experts.
    
    Args:
        model: The Mixtral model
        
    Returns:
        Dictionary mapping layer names to their modules
    """
    router_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'gate' in name.lower():
            router_layers[name] = module
    return router_layers


def _add_module_by_name(model, name: str, new_module: nn.Module):
    """
    Replace a submodule in the model by its dotted name.
    
    Args:
        model: The parent model
        name: Dotted name of the module to replace (e.g., "model.layers.0.block_sparse_moe.gate")
        new_module: The new module to insert
    """
    parts = name.split('.')
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def replace_router_with_lora(
    model,
    rank: int = 8,
    lora_alpha: float = 16.0,
    device: Optional[torch.device] = None
) -> Dict[str, LoraLinear]:
    """
    Replace all router layers in a Mixtral model with LoRA adapters.
    The base weights remain frozen.
    
    Args:
        model: The Mixtral model
        rank: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        device: Device to place the LoRA layers on
        
    Returns:
        Dictionary mapping layer names to their LoraLinear modules
    """
    router_layers = get_router_layers(model)
    lora_modules = {}
    
    for name, module in router_layers.items():
        lora_linear = LoraLinear(module, rank=rank, lora_alpha=lora_alpha)
        if device is not None:
            lora_linear = lora_linear.to(device)
        _add_module_by_name(model, name, lora_linear)
        lora_modules[name] = lora_linear
        
    return lora_modules


def extract_lora_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Extract LoRA parameters from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file (omni_parameters.pth)
        
    Returns:
        Dictionary mapping LoRA parameter names to their tensors
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    lora_params = {}
    
    # The checkpoint is organized by layer index
    for layer_idx, layer_params in checkpoint.items():
        if not isinstance(layer_params, dict):
            continue
        for param_name, param_value in layer_params.items():
            if 'lora_A' in param_name or 'lora_B' in param_name:
                # Reconstruct full parameter name
                full_name = f"layer_{layer_idx}.{param_name}"
                lora_params[full_name] = param_value
                
    return lora_params


def load_lora_weights(
    model,
    checkpoint_path: str,
    rank: int = 8,
    lora_alpha: float = 16.0,
    device: Optional[torch.device] = None,
    strict: bool = False
) -> Dict[str, LoraLinear]:
    """
    Load LoRA weights from a checkpoint and apply to Mixtral model's router layers.
    This function:
    1. Replaces router layers with LoraLinear modules
    2. Loads LoRA weights from checkpoint
    3. Keeps base model weights frozen
    
    Args:
        model: The Mixtral model
        checkpoint_path: Path to the omni_parameters.pth checkpoint
        rank: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        device: Device to place the LoRA layers on
        strict: Whether to raise error if checkpoint parameters don't match
        
    Returns:
        Dictionary mapping layer names to their LoraLinear modules
    """
    # First, replace router layers with LoRA adapters
    lora_modules = replace_router_with_lora(model, rank=rank, lora_alpha=lora_alpha, device=device)
    
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Using initialized LoRA weights.")
        return lora_modules
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Map checkpoint parameters to model layers
    loaded_count = 0
    missing_params = []
    
    for layer_idx, layer_params in checkpoint.items():
        if not isinstance(layer_params, dict):
            continue
            
        for param_name, param_value in layer_params.items():
            if 'lora_A' not in param_name and 'lora_B' not in param_name:
                continue
                
            # Find corresponding LoRA module
            # param_name format: "block_sparse_moe.gate.lora_A" or similar
            matched = False
            for module_name, lora_module in lora_modules.items():
                # Check if this parameter belongs to this layer
                if f"layers.{layer_idx}." in module_name or f"layers[{layer_idx}]" in module_name:
                    if 'gate' in module_name:
                        if 'lora_A' in param_name:
                            lora_module.lora_A.data = param_value.to(lora_module.lora_A.device)
                            loaded_count += 1
                            matched = True
                        elif 'lora_B' in param_name:
                            lora_module.lora_B.data = param_value.to(lora_module.lora_B.device)
                            loaded_count += 1
                            matched = True
                        break
            
            # Alternative matching: try to match by layer index from checkpoint structure
            if not matched:
                for module_name, lora_module in lora_modules.items():
                    # Extract layer index from module name
                    import re
                    match = re.search(r'layers\.(\d+)\.', module_name)
                    if match and int(match.group(1)) == layer_idx:
                        if 'gate' in module_name or 'gate' in param_name:
                            if 'lora_A' in param_name:
                                lora_module.lora_A.data = param_value.to(lora_module.lora_A.device)
                                loaded_count += 1
                                matched = True
                            elif 'lora_B' in param_name:
                                lora_module.lora_B.data = param_value.to(lora_module.lora_B.device)
                                loaded_count += 1
                                matched = True
                            break
            
            if not matched:
                missing_params.append(f"layer_{layer_idx}.{param_name}")
    
    if loaded_count > 0:
        print(f"Successfully loaded {loaded_count} LoRA parameters from checkpoint")
    
    if missing_params and strict:
        raise ValueError(f"Failed to load LoRA parameters: {missing_params}")
    elif missing_params:
        print(f"Warning: {len(missing_params)} LoRA parameters not matched")
        
    return lora_modules


def freeze_base_model(model) -> None:
    """
    Freeze all parameters in the base model except LoRA parameters.
    This ensures that during evaluation, only LoRA adapters are active
    and base model weights remain unchanged.
    
    Args:
        model: The model to freeze
    """
    for name, param in model.named_parameters():
        if 'lora_A' not in name and 'lora_B' not in name:
            param.requires_grad = False
        else:
            # LoRA parameters can be trainable, but for evaluation we set them to eval mode
            param.requires_grad = False


def get_lora_parameter_count(model) -> Tuple[int, int]:
    """
    Get the count of LoRA parameters and total parameters in the model.
    
    Args:
        model: The model
        
    Returns:
        Tuple of (lora_param_count, total_param_count)
    """
    lora_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if 'lora_A' in name or 'lora_B' in name:
            lora_params += param.numel()
            
    return lora_params, total_params


def prepare_mixtral_for_lora_eval(
    model,
    model_name: str,
    checkpoint_path: Optional[str] = None,
    rank: int = 8,
    lora_alpha: float = 16.0,
    device: Optional[torch.device] = None
) -> bool:
    """
    Main entry point for preparing a Mixtral model for LoRA evaluation.
    
    Args:
        model: The model to prepare
        model_name: Name of the model (used to detect if it's Mixtral)
        checkpoint_path: Path to LoRA checkpoint (optional)
        rank: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        device: Device to use
        
    Returns:
        True if LoRA was applied, False otherwise
    """
    if not is_mixtral_model(model_name):
        return False
    
    # Replace router layers with LoRA
    if checkpoint_path:
        lora_modules = load_lora_weights(
            model, 
            checkpoint_path, 
            rank=rank, 
            lora_alpha=lora_alpha, 
            device=device
        )
    else:
        lora_modules = replace_router_with_lora(
            model,
            rank=rank,
            lora_alpha=lora_alpha,
            device=device
        )
    
    # Freeze base model
    freeze_base_model(model)
    
    # Print LoRA parameter statistics
    lora_count, total_count = get_lora_parameter_count(model)
    print(f"LoRA parameters: {lora_count:,} / Total parameters: {total_count:,}")
    print(f"LoRA ratio: {100 * lora_count / total_count:.4f}%")
    
    return True
