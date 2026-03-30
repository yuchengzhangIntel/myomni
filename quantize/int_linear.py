import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import cast
from quantize.quantizer import UniformAffineQuantizer






class QuantLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        disable_input_quant=False,
        use_linear_lora: bool = False,
        linear_lora_r: int = 16,
        linear_lora_alpha: float = 16.0,
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_buffer('weight',org_module.weight)
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params,shape=org_module.weight.shape)
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False
        self.use_linear_lora = use_linear_lora
        self.linear_lora_r = linear_lora_r
        self.linear_lora_alpha = linear_lora_alpha
        self.linear_lora_scaling = 0.0

        if self.use_linear_lora:
            if self.linear_lora_r <= 0:
                raise ValueError("linear_lora_r must be positive when use_linear_lora=True")
            self.linear_lora_scaling = self.linear_lora_alpha / self.linear_lora_r
            self.lora_A = nn.Parameter(torch.empty(
                self.linear_lora_r,
                self.in_features,
                device=org_module.weight.device,
                dtype=org_module.weight.dtype,
            ))
            self.lora_B = nn.Parameter(torch.zeros(
                self.out_features,
                self.linear_lora_r,
                device=org_module.weight.device,
                dtype=org_module.weight.dtype,
            ))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter('lora_A', None)
            self.register_parameter('lora_B', None)

    def get_lora_delta(self):
        if not self.use_linear_lora or self.lora_A is None or self.lora_B is None:
            return None
        return self.linear_lora_scaling * (self.lora_B @ self.lora_A)

    def get_effective_weight(self):
        delta = self.get_lora_delta()
        if delta is None:
            return self.weight
        return self.weight + delta

    def get_lora_parameters(self):
        if not self.use_linear_lora:
            return []
        return [self.lora_A, self.lora_B]

    @torch.no_grad()
    def merge_lora(self):
        if not self.use_linear_lora:
            return False

        delta = self.get_lora_delta()
        if delta is None:
            return False
        delta = cast(torch.Tensor, delta)

        self.weight = self.weight + delta.detach()
        self.lora_A = None
        self.lora_B = None
        self.use_linear_lora = False
        self.linear_lora_scaling = 0.0
        return True

    
    
    def forward(self, input: torch.Tensor):
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        elif self.use_weight_quant:
            weight = self.weight_quantizer(self.get_effective_weight())
            bias = self.bias
        else:
            weight = self.get_effective_weight()
            bias = self.bias

        if self.use_act_quant and not self.disable_input_quant:
            if self.act_quantizer is None:
                raise RuntimeError("act_quantizer is not initialized")
            input = self.act_quantizer(input)

        weight_tensor = cast(torch.Tensor, weight)
        bias_tensor = cast(torch.Tensor | None, bias)
        out = F.linear(input, weight_tensor, bias_tensor, **self.fwd_kwargs)


        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
