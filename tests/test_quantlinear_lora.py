import torch
import torch.nn as nn
from types import SimpleNamespace

from quantize.int_linear import QuantLinear
from quantize.utils import smooth_and_quant_inplace


def build_quantlinear(use_linear_lora=True):
    linear = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]))

    module = QuantLinear(
        linear,
        {"n_bits": 16},
        {"n_bits": 16},
        use_linear_lora=use_linear_lora,
        linear_lora_r=2,
        linear_lora_alpha=2.0,
    )
    return module


def test_quantlinear_effective_weight_includes_lora_delta():
    module = build_quantlinear()
    with torch.no_grad():
        module.lora_A.copy_(torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]))
        module.lora_B.copy_(torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
        ]))

    expected_delta = torch.tensor([
        [1.0, 2.0, 0.0],
        [3.0, 4.0, 0.0],
    ])
    expected_weight = module.weight + expected_delta

    assert torch.allclose(module.get_lora_delta(), expected_delta)
    assert torch.allclose(module.get_effective_weight(), expected_weight)


def test_smooth_and_quant_inplace_merges_lora_once():
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = build_quantlinear()

    wrapper = Wrapper()
    with torch.no_grad():
        wrapper.linear.lora_A.copy_(torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]))
        wrapper.linear.lora_B.copy_(torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
        ]))

    args = SimpleNamespace(let=False)
    input_tensor = torch.tensor([[1.0, 1.0, 1.0]])
    expected_output = nn.functional.linear(input_tensor, wrapper.linear.get_effective_weight())

    smooth_and_quant_inplace(wrapper, args, isllama=False)
    actual_output = wrapper.linear(input_tensor)

    assert not wrapper.linear.use_linear_lora
    assert wrapper.linear.lora_A is None
    assert wrapper.linear.lora_B is None
    assert torch.allclose(actual_output, expected_output)
