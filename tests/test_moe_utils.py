import os
import sys
import gc
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantize.moe_utils import (  # noqa: E402
    QuantizedPackedExperts,
    compute_expert_down_proj_output,
    compute_router_scores,
    get_moe_top_k,
    pin_cpu_tensor,
    select_top_n_experts,
)
import utils as root_utils_module  # noqa: E402
import quantize.block_evaluator as block_evaluator_module  # noqa: E402
import quantize.omniquant as omniquant_module  # noqa: E402
from quantize.block_evaluator import evaluate_block_loss_modes, update_block_parameters_with_loss  # noqa: E402
from quantize.omniquant import (  # noqa: E402
    build_block_update_param_groups,
    build_router_calibration_param_groups,
    compute_moe_self_supervision_loss,
    should_keep_layer_full_precision,
    train_decoupled_moe_layer,
    train_stage0_attention_alignment,
)


def build_args():
    return SimpleNamespace(
        weight_quant_params={"n_bits": 16},
        act_quant_params={"n_bits": 16},
        use_linear_lora=False,
        linear_lora_r=4,
        linear_lora_alpha=4.0,
    )


class FakePackedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 2
        self.hidden_dim = 3
        self.intermediate_dim = 2
        self.act_fn = F.silu
        self.gate_up_proj = nn.Parameter(torch.tensor([
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
            ],
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.5],
            ],
        ]))
        self.down_proj = nn.Parameter(torch.tensor([
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
            ],
            [
                [0.5, 0.0],
                [0.0, 0.5],
                [1.0, 1.0],
            ],
        ]))


class FakeQwen2Expert(nn.Module):
    def __init__(self, gate_weight, up_weight, down_weight):
        super().__init__()
        hidden_size = gate_weight.shape[1]
        intermediate_size = gate_weight.shape[0]
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = F.silu

        with torch.no_grad():
            self.gate_proj.weight.copy_(gate_weight)
            self.up_proj.weight.copy_(up_weight)
            self.down_proj.weight.copy_(down_weight)

    def forward(self, hidden_states):
        gated = self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.down_proj(gated)


def test_quantized_packed_experts_matches_packed_forward():
    packed = FakePackedExperts()
    quantized = QuantizedPackedExperts(packed, build_args())

    hidden_states = torch.tensor([
        [1.0, 2.0, 0.0],
        [0.0, 1.0, 3.0],
    ])
    top_k_index = torch.tensor([[0, 1], [1, 0]])
    top_k_weights = torch.tensor([[0.7, 0.3], [0.6, 0.4]])

    expected = torch.zeros_like(hidden_states)
    for token_idx in range(hidden_states.shape[0]):
        for top_pos in range(top_k_index.shape[1]):
            expert_idx = int(top_k_index[token_idx, top_pos].item())
            expert_output = compute_expert_down_proj_output(
                packed,
                expert_idx,
                hidden_states[token_idx : token_idx + 1],
            )
            expected[token_idx] += top_k_weights[token_idx, top_pos] * expert_output.squeeze(0)

    actual = quantized(hidden_states, top_k_index, top_k_weights)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_should_keep_layer_full_precision_respects_max_train_layers():
    assert should_keep_layer_full_precision(layer_idx=0, max_train_layers=-1) is False
    assert should_keep_layer_full_precision(layer_idx=0, max_train_layers=1) is False
    assert should_keep_layer_full_precision(layer_idx=1, max_train_layers=1) is True
    assert should_keep_layer_full_precision(layer_idx=3, max_train_layers=2) is True


def test_router_score_selection_normalizes_top_n_weights():
    class TupleGate(nn.Module):
        def forward(self, hidden_states):
            probs = torch.tensor([
                [0.1, 0.3, 0.6],
                [0.5, 0.2, 0.3],
            ], dtype=torch.float32, device=hidden_states.device)
            return probs, probs[:, :2], torch.tensor([[2, 1], [0, 2]], device=hidden_states.device)

    moe = SimpleNamespace(gate=TupleGate())
    hidden_states = torch.randn(2, 3)
    router_scores = compute_router_scores(moe, hidden_states)
    top_indices, top_weights = select_top_n_experts(router_scores, top_n=2)

    assert top_indices.shape == (2, 2)
    assert torch.allclose(top_weights.sum(dim=-1), torch.ones(2))
    assert top_indices[0].tolist() == [2, 1]
    assert top_indices[1].tolist() == [0, 2]


def test_get_moe_top_k_reads_router_top_k_when_block_has_no_top_k():
    class FakeTopKRouter(nn.Module):
        def __init__(self):
            super().__init__()
            self.top_k = 4

        def forward(self, hidden_states):
            probs = torch.softmax(torch.ones(hidden_states.shape[0], 8, device=hidden_states.device), dim=-1)
            indices = torch.zeros(hidden_states.shape[0], self.top_k, dtype=torch.long, device=hidden_states.device)
            return probs, probs[:, : self.top_k], indices

    moe = SimpleNamespace(gate=FakeTopKRouter())

    assert get_moe_top_k(moe) == 4


def test_compute_expert_down_proj_output_supports_modulelist_experts():
    experts = nn.ModuleList([
        FakeQwen2Expert(
            gate_weight=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            up_weight=torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]),
            down_weight=torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.25, 0.25]]),
        ),
        FakeQwen2Expert(
            gate_weight=torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            up_weight=torch.tensor([[0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]),
            down_weight=torch.tensor([[0.5, 0.0], [0.0, 0.5], [1.0, 1.0]]),
        ),
    ])
    hidden_states = torch.tensor([[0.0, 1.0, 3.0]], dtype=torch.float32)

    actual = compute_expert_down_proj_output(experts, 1, hidden_states)
    expected = experts[1](hidden_states)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_moe_self_supervision_loss_handles_experts_and_shared_expert():
    class ZeroAttention(nn.Module):
        def forward(self, hidden_states, **kwargs):
            return (torch.zeros_like(hidden_states), None)

    class IdentityNorm(nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    class DummyGate(nn.Module):
        def forward(self, hidden_states):
            probs = torch.full((hidden_states.shape[0], 2), 0.5, device=hidden_states.device)
            return probs, probs, torch.zeros((hidden_states.shape[0], 1), dtype=torch.long, device=hidden_states.device)

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = IdentityNorm()
            self.post_attention_layernorm = IdentityNorm()
            self.self_attn = ZeroAttention()
            self.mlp = nn.Module()
            self.mlp.gate = DummyGate()
            self.mlp.experts = QuantizedPackedExperts(FakePackedExperts(), build_args())
            self.mlp.shared_expert = nn.Linear(3, 3, bias=False)
            with torch.no_grad():
                self.mlp.shared_expert.weight.copy_(torch.eye(3))

    qlayer = FakeLayer()
    quant_inputs = torch.tensor([[[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]]])
    sample_inputs = quant_inputs[0]

    expert_zero = compute_expert_down_proj_output(qlayer.mlp.experts, 0, sample_inputs[:1])
    expert_one = compute_expert_down_proj_output(qlayer.mlp.experts, 1, sample_inputs[1:2])
    shared_labels = qlayer.mlp.shared_expert(sample_inputs)

    batch_label_cache = [{
        "expert_labels": {
            0: {
                "token_idx": pin_cpu_tensor(torch.tensor([0], dtype=torch.long)),
                "weights": pin_cpu_tensor(torch.tensor([1.0], dtype=torch.float32)),
                "labels": pin_cpu_tensor(expert_zero),
            },
            1: {
                "token_idx": pin_cpu_tensor(torch.tensor([1], dtype=torch.long)),
                "weights": pin_cpu_tensor(torch.tensor([1.0], dtype=torch.float32)),
                "labels": pin_cpu_tensor(expert_one),
            },
        },
        "shared_labels": pin_cpu_tensor(shared_labels),
    }]

    loss = compute_moe_self_supervision_loss(
        qlayer,
        quant_inputs,
        batch_label_cache,
        layer_kwargs={},
        attention_mask=None,
        position_ids=None,
        use_router_weight_in_loss=True,
    )

    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-7)


def test_moe_self_supervision_loss_accepts_precomputed_mlp_inputs():
    class ZeroAttention(nn.Module):
        def forward(self, hidden_states, **kwargs):
            return (torch.zeros_like(hidden_states), None)

    class IdentityNorm(nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    class DummyGate(nn.Module):
        def forward(self, hidden_states):
            probs = torch.full((hidden_states.shape[0], 2), 0.5, device=hidden_states.device)
            return probs, probs, torch.zeros((hidden_states.shape[0], 1), dtype=torch.long, device=hidden_states.device)

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = IdentityNorm()
            self.post_attention_layernorm = IdentityNorm()
            self.self_attn = ZeroAttention()
            self.mlp = nn.Module()
            self.mlp.gate = DummyGate()
            self.mlp.experts = QuantizedPackedExperts(FakePackedExperts(), build_args())
            self.mlp.shared_expert = nn.Linear(3, 3, bias=False)
            with torch.no_grad():
                self.mlp.shared_expert.weight.copy_(torch.eye(3))

    qlayer = FakeLayer()
    mlp_inputs = torch.tensor([[[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]]], requires_grad=False)
    sample_inputs = mlp_inputs[0]

    expert_zero = compute_expert_down_proj_output(qlayer.mlp.experts, 0, sample_inputs[:1])
    expert_one = compute_expert_down_proj_output(qlayer.mlp.experts, 1, sample_inputs[1:2])
    shared_labels = qlayer.mlp.shared_expert(sample_inputs)

    batch_label_cache = [{
        "expert_labels": {
            0: {
                "token_idx": pin_cpu_tensor(torch.tensor([0], dtype=torch.long)),
                "weights": pin_cpu_tensor(torch.tensor([1.0], dtype=torch.float32)),
                "labels": pin_cpu_tensor(expert_zero),
            },
            1: {
                "token_idx": pin_cpu_tensor(torch.tensor([1], dtype=torch.long)),
                "weights": pin_cpu_tensor(torch.tensor([1.0], dtype=torch.float32)),
                "labels": pin_cpu_tensor(expert_one),
            },
        },
        "shared_labels": pin_cpu_tensor(shared_labels),
    }]

    loss = compute_moe_self_supervision_loss(
        qlayer,
        quant_inputs=None,
        batch_label_cache=batch_label_cache,
        layer_kwargs={},
        attention_mask=None,
        position_ids=None,
        use_router_weight_in_loss=False,
        precomputed_mlp_inputs=mlp_inputs.detach(),
    )

    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-7)


def test_moe_self_supervision_loss_reduces_per_expert_before_aggregation():
    class ZeroAttention(nn.Module):
        def forward(self, hidden_states, **kwargs):
            return (torch.zeros_like(hidden_states), None)

    class IdentityNorm(nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    class IdentityExpert(nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = IdentityNorm()
            self.post_attention_layernorm = IdentityNorm()
            self.self_attn = ZeroAttention()
            self.mlp = nn.Module()
            self.mlp.experts = nn.ModuleList([IdentityExpert(), IdentityExpert()])

    qlayer = FakeLayer()
    mlp_inputs = torch.tensor([
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [3.0, 0.0],
        ]
    ])

    batch_label_cache = [{
        "expert_labels": {
            0: {
                "token_idx": pin_cpu_tensor(torch.tensor([0, 1], dtype=torch.long)),
                "weights": pin_cpu_tensor(torch.tensor([1.0, 1.0], dtype=torch.float32)),
                "labels": pin_cpu_tensor(torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)),
            },
            1: {
                "token_idx": pin_cpu_tensor(torch.tensor([2], dtype=torch.long)),
                "weights": pin_cpu_tensor(torch.tensor([1.0], dtype=torch.float32)),
                "labels": pin_cpu_tensor(torch.tensor([[0.0, 0.0]], dtype=torch.float32)),
            },
        },
        "shared_labels": None,
    }]

    loss = compute_moe_self_supervision_loss(
        qlayer,
        quant_inputs=None,
        batch_label_cache=batch_label_cache,
        layer_kwargs={},
        attention_mask=None,
        position_ids=None,
        use_router_weight_in_loss=False,
        precomputed_mlp_inputs=mlp_inputs,
    )

    assert torch.allclose(loss, torch.tensor(5.0), atol=1e-7)


def test_moe_self_supervision_loss_aggregates_same_expert_tokens_across_batch():
    class ZeroAttention(nn.Module):
        def forward(self, hidden_states, **kwargs):
            return (torch.zeros_like(hidden_states), None)

    class IdentityNorm(nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    class IdentityExpert(nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = IdentityNorm()
            self.post_attention_layernorm = IdentityNorm()
            self.self_attn = ZeroAttention()
            self.mlp = nn.Module()
            self.mlp.experts = nn.ModuleList([IdentityExpert()])

    qlayer = FakeLayer()
    mlp_inputs = torch.tensor([
        [[1.0], [0.0]],
        [[3.0], [3.0]],
    ])

    batch_label_cache = [
        {
            "expert_labels": {
                0: {
                    "token_idx": pin_cpu_tensor(torch.tensor([0], dtype=torch.long)),
                    "weights": pin_cpu_tensor(torch.tensor([1.0], dtype=torch.float32)),
                    "labels": pin_cpu_tensor(torch.tensor([[0.0]], dtype=torch.float32)),
                },
            },
            "shared_labels": None,
        },
        {
            "expert_labels": {
                0: {
                    "token_idx": pin_cpu_tensor(torch.tensor([0, 1], dtype=torch.long)),
                    "weights": pin_cpu_tensor(torch.tensor([1.0, 1.0], dtype=torch.float32)),
                    "labels": pin_cpu_tensor(torch.tensor([[0.0], [0.0]], dtype=torch.float32)),
                },
            },
            "shared_labels": None,
        },
    ]

    loss = compute_moe_self_supervision_loss(
        qlayer,
        quant_inputs=None,
        batch_label_cache=batch_label_cache,
        layer_kwargs={},
        attention_mask=None,
        position_ids=None,
        use_router_weight_in_loss=False,
        precomputed_mlp_inputs=mlp_inputs,
    )

    assert torch.allclose(loss, torch.tensor(19.0 / 3.0), atol=1e-7)


def test_moe_self_supervision_loss_backpropagates_to_attention_without_precomputed_inputs():
    class LearnableAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.proj.weight.copy_(torch.eye(2))

        def forward(self, hidden_states, **kwargs):
            return self.proj(hidden_states), None

    class IdentityNorm(nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    class IdentityExpert(nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = IdentityNorm()
            self.post_attention_layernorm = IdentityNorm()
            self.self_attn = LearnableAttention()
            self.mlp = nn.Module()
            self.mlp.experts = nn.ModuleList([IdentityExpert()])

    qlayer = FakeLayer()
    quant_inputs = torch.tensor([[[1.0, 2.0]]], dtype=torch.float32)
    batch_label_cache = [{
        "expert_labels": {
            0: {
                "token_idx": pin_cpu_tensor(torch.tensor([0], dtype=torch.long)),
                "weights": pin_cpu_tensor(torch.tensor([1.0], dtype=torch.float32)),
                "labels": pin_cpu_tensor(torch.tensor([[0.0, 0.0]], dtype=torch.float32)),
            },
        },
        "shared_labels": None,
    }]

    loss = compute_moe_self_supervision_loss(
        qlayer,
        quant_inputs=quant_inputs,
        batch_label_cache=batch_label_cache,
        layer_kwargs={},
        attention_mask=None,
        position_ids=None,
        use_router_weight_in_loss=False,
    )
    loss.backward()

    assert qlayer.self_attn.proj.weight.grad is not None
    assert torch.count_nonzero(qlayer.self_attn.proj.weight.grad).item() > 0


def test_build_block_update_param_groups_respects_scope_flags():
    class FakeQuantLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.bound_factor = nn.Parameter(torch.tensor(1.0))
            self.lora_A = nn.Parameter(torch.tensor([[1.0]], dtype=torch.float32))
            self.lora_B = nn.Parameter(torch.tensor([[1.0]], dtype=torch.float32))

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.q_proj = FakeQuantLinear()
            self.mlp = nn.Module()
            self.mlp.gate = nn.Linear(1, 1, bias=False)
            self.mlp.shared_expert_gate = nn.Linear(1, 1, bias=False)

    qlayer = FakeLayer()
    args = SimpleNamespace(
        use_linear_lora=True,
        lwc_lr=1e-2,
        linear_lora_lr=1e-4,
        wd=0.01,
        gate_lora_lr=2e-4,
        shared_gate_lr=3e-4,
        stage3_update_attn=False,
        stage3_update_router=False,
    )

    selected_params, param_groups = build_block_update_param_groups(qlayer, args)
    assert selected_params == []
    assert param_groups == []

    args.stage3_update_attn = True
    selected_params, param_groups = build_block_update_param_groups(qlayer, args)
    selected_ids = {id(param) for param in selected_params}
    assert id(qlayer.self_attn.q_proj.bound_factor) in selected_ids
    assert id(qlayer.self_attn.q_proj.lora_A) in selected_ids
    assert id(qlayer.self_attn.q_proj.lora_B) in selected_ids
    assert id(qlayer.mlp.gate.weight) not in selected_ids
    assert len(param_groups) == 2

    args.stage3_update_attn = False
    args.stage3_update_router = True
    selected_params, param_groups = build_block_update_param_groups(qlayer, args)
    selected_ids = {id(param) for param in selected_params}
    assert id(qlayer.self_attn.q_proj.bound_factor) not in selected_ids
    assert id(qlayer.mlp.gate.weight) in selected_ids
    assert id(qlayer.mlp.shared_expert_gate.weight) in selected_ids
    assert len(param_groups) == 2


def test_build_router_calibration_param_groups_respects_attention_flag():
    class FakeQuantLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.bound_factor = nn.Parameter(torch.tensor(1.0))
            self.lora_A = nn.Parameter(torch.tensor([[1.0]], dtype=torch.float32))
            self.lora_B = nn.Parameter(torch.tensor([[1.0]], dtype=torch.float32))

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.q_proj = FakeQuantLinear()
            self.mlp = nn.Module()
            self.mlp.gate = nn.Linear(1, 1, bias=False)

    qlayer = FakeLayer()
    args = SimpleNamespace(
        use_linear_lora=True,
        stage2_router_lr=1e-3,
        lwc_lr=1e-2,
        linear_lora_lr=1e-4,
        wd=0.01,
        stage2_update_attn=False,
    )

    selected_params, param_groups = build_router_calibration_param_groups(qlayer, args)
    selected_ids = {id(param) for param in selected_params}
    assert id(qlayer.mlp.gate.weight) in selected_ids
    assert id(qlayer.self_attn.q_proj.bound_factor) not in selected_ids
    assert len(param_groups) == 1

    args.stage2_update_attn = True
    selected_params, param_groups = build_router_calibration_param_groups(qlayer, args)
    selected_ids = {id(param) for param in selected_params}
    assert id(qlayer.mlp.gate.weight) in selected_ids
    assert id(qlayer.self_attn.q_proj.bound_factor) in selected_ids
    assert id(qlayer.self_attn.q_proj.lora_A) in selected_ids
    assert id(qlayer.self_attn.q_proj.lora_B) in selected_ids
    assert len(param_groups) == 3


def test_train_stage0_attention_alignment_updates_only_attention():
    class DummyLogger:
        def info(self, *_args, **_kwargs):
            return None

        def warning(self, *_args, **_kwargs):
            return None

    class DummyAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.bound_factor = nn.Parameter(torch.tensor(0.5))

        def forward(self, hidden_states, **kwargs):
            # Quantized attention output depends on the trainable bound_factor so
            # the alignment loss has a real gradient w.r.t. attention params.
            return hidden_states + self.bound_factor

    class DummyMLPExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.bound_factor = nn.Parameter(torch.tensor(0.0))

    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = nn.Identity()
            self.post_attention_layernorm = nn.Identity()
            self.self_attn = DummyAttn()
            self.mlp = nn.Module()
            self.mlp.experts = DummyMLPExpert()

        def forward(self, hidden_states, **kwargs):
            return hidden_states

    # FP teacher path: attention contributes 0.5 to the residual MLP input.
    layer = DummyLayer()
    qlayer = DummyLayer()
    qlayer.self_attn.bound_factor.data.fill_(0.0)

    args = SimpleNamespace(
        nsamples=2,
        batch_size=1,
        lwc_lr=1e-1,
        linear_lora_lr=1e-4,
        wd=0.0,
        max_grad_norm=None,
        use_linear_lora=False,
        stage0_attn_epochs=3,
    )

    original_smooth = omniquant_module.smooth_and_quant_temporary
    original_clear = omniquant_module.clear_temp_variable
    original_set_quant = omniquant_module.set_quant_state
    try:
        omniquant_module.smooth_and_quant_temporary = lambda *a, **k: None
        omniquant_module.clear_temp_variable = lambda *a, **k: None
        omniquant_module.set_quant_state = lambda *a, **k: None

        # stage0 disabled -> returns None, no updates
        args.stage0_attn_epochs = 0
        assert train_stage0_attention_alignment(
            layer, qlayer, args, DummyLogger(), 0,
            torch.zeros(2, 1, 1), torch.zeros(2, 1, 1),
            {}, None, None, None, nullcontext, False,
        ) is None

        # stage0 enabled -> trains attn toward FP target (0.5), expert stays frozen
        args.stage0_attn_epochs = 3
        expert_before = qlayer.mlp.experts.bound_factor.detach().clone()
        loss = train_stage0_attention_alignment(
            layer, qlayer, args, DummyLogger(), 0,
            torch.zeros(2, 1, 1), torch.zeros(2, 1, 1),
            {}, None, None, None, nullcontext, False,
        )
        assert loss is not None
        # attention param moved toward the FP attention output (0.5)
        assert qlayer.self_attn.bound_factor.item() > 0.0
        # expert param untouched
        assert torch.equal(qlayer.mlp.experts.bound_factor.detach(), expert_before)
        # requires_grad restored after stage0
        assert qlayer.mlp.experts.bound_factor.requires_grad
    finally:
        omniquant_module.smooth_and_quant_temporary = original_smooth
        omniquant_module.clear_temp_variable = original_clear
        omniquant_module.set_quant_state = original_set_quant


def test_evaluate_block_loss_modes_reports_student_only():
    class DummyLogger:
        def info(self, *_args, **_kwargs):
            return None

    class IdentityLayer(nn.Module):
        def forward(self, hidden_states, **kwargs):
            return hidden_states

    args = SimpleNamespace(nsamples=2, batch_size=1, let=False)
    qlayer = IdentityLayer()
    quant_inputs = torch.tensor([[[1.0]], [[2.0]]], dtype=torch.float32)
    fp_targets = quant_inputs.clone()

    results = evaluate_block_loss_modes(
        qlayer=qlayer,
        args=args,
        loss_func=nn.MSELoss(),
        quant_inputs=quant_inputs,
        fp_targets=fp_targets,
        fp_targets_aug=None,
        layer_kwargs={},
        attention_mask_batch=None,
        position_ids=None,
        traincast=nullcontext,
        logger=DummyLogger(),
        layer_idx=0,
        epoch_idx=0,
        smooth_is_llama=False,
    )

    assert list(results.keys()) == ["student"]


def test_update_block_parameters_with_loss_skips_batches_with_nonfinite_norm():
    class DummyLogger:
        def __init__(self):
            self.infos = []
            self.warnings = []

        def info(self, message, *args, **kwargs):
            self.infos.append(message)

        def warning(self, message, *args, **kwargs):
            self.warnings.append(message)

    class ScaledIdentityLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, hidden_states, **kwargs):
            return hidden_states * self.scale

    class FakeScaler:
        def __init__(self):
            self.calls = 0

        def __call__(self, loss, optimizer, clip_grad=None, parameters=None, return_metadata=False, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return torch.tensor(float("inf")), True, "nonfinite_grad_norm"

            loss.backward()
            optimizer.step()
            return torch.tensor(0.25), False, None

    qlayer = ScaledIdentityLayer()
    optimizer = torch.optim.SGD([qlayer.scale], lr=0.1)
    logger = DummyLogger()
    quant_inputs = torch.tensor([[[1.0]], [[2.0]]], dtype=torch.float32)
    fp_targets = torch.zeros_like(quant_inputs)

    final_total = update_block_parameters_with_loss(
        qlayer=qlayer,
        args=SimpleNamespace(nsamples=2, batch_size=1, let=False),
        optimizer=optimizer,
        loss_scaler=FakeScaler(),
        clip_parameters=[qlayer.scale],
        loss_func=nn.MSELoss(),
        quant_inputs=quant_inputs,
        fp_targets=fp_targets,
        fp_targets_aug=None,
        teacher_router_labels=None,
        aux_enabled=False,
        aux_weight=0.0,
        aux_topk=1,
        layer_kwargs={},
        attention_mask_batch=None,
        position_ids=None,
        traincast=nullcontext,
        logger=logger,
        layer_idx=0,
        smooth_is_llama=False,
        update_epochs=1,
        clip_grad=None,
    )

    assert abs(final_total - 4.0) < 1e-6
    assert any("optimizer update skipped due to nonfinite_grad_norm" in msg for msg in logger.warnings)


def test_native_scaler_updates_grad_scaler_state_when_large_grad_skips_step():
    class FakeScaledLoss:
        def __init__(self, loss):
            self.loss = loss

        def backward(self, create_graph=False, retain_graph=False):
            self.loss.backward(create_graph=create_graph, retain_graph=retain_graph)

    class FakeGradScaler:
        def __init__(self):
            self.scale_value = 128.0
            self.stage = "ready"
            self.step_calls = 0
            self.update_args = []

        def scale(self, loss):
            return FakeScaledLoss(loss)

        def unscale_(self, optimizer):
            if self.stage != "ready":
                raise RuntimeError("unscale_() has already been called on this optimizer since the last update().")
            self.stage = "unscaled"

        def step(self, optimizer):
            if self.stage != "unscaled":
                raise RuntimeError("step() called before unscale_().")
            self.step_calls += 1
            self.stage = "stepped"

        def update(self, new_scale=None):
            self.update_args.append(new_scale)
            self.stage = "ready"

        def get_scale(self):
            return self.scale_value

    scaler = root_utils_module.NativeScalerWithGradNormCount(use_grad_scaler=False, max_grad_norm_for_update=1.0)
    fake_scaler = FakeGradScaler()
    scaler._use_grad_scaler = True
    scaler._scaler = fake_scaler

    original_get_grad_norm = root_utils_module.ampscaler_get_grad_norm
    try:
        root_utils_module.ampscaler_get_grad_norm = lambda parameters, norm_type=2.0: torch.tensor(1e5)

        parameter = nn.Parameter(torch.tensor(2.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)

        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = parameter.square()
            norm, skipped_update, skip_reason = scaler(
                loss,
                optimizer,
                parameters=[parameter],
                return_metadata=True,
            )

            assert skipped_update is True
            assert skip_reason == "grad_norm_too_large"
            assert float(norm.detach().item()) == 1e5
            assert parameter.grad is None

        assert fake_scaler.step_calls == 0
        assert fake_scaler.update_args == [fake_scaler.scale_value, fake_scaler.scale_value]
    finally:
        root_utils_module.ampscaler_get_grad_norm = original_get_grad_norm


def test_train_decoupled_moe_layer_clears_temp_weights_when_moe_stage_raises():
    class DummyLogger:
        def __init__(self):
            self.infos = []
            self.warnings = []

        def info(self, message, *args, **kwargs):
            self.infos.append(message)

        def warning(self, message, *args, **kwargs):
            self.warnings.append(message)

    class DummyGate(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1, 1))
            self.top_k = 1
            self.out_features = 1

        def forward(self, hidden_states):
            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            logits = torch.ones(flat.shape[0], self.out_features, device=hidden_states.device)
            return logits, logits, torch.zeros(flat.shape[0], 1, dtype=torch.long, device=hidden_states.device)

    class DummyQuantModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.bound_factor = nn.Parameter(torch.tensor(0.0))

        def forward(self, hidden_states, **kwargs):
            return hidden_states

    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = nn.Identity()
            self.post_attention_layernorm = nn.Identity()
            self.self_attn = nn.Module()
            self.self_attn.q_proj = DummyQuantModule()
            self.mlp = nn.Module()
            self.mlp.gate = DummyGate()
            self.mlp.experts = DummyQuantModule()

        def forward(self, hidden_states, **kwargs):
            return hidden_states

    class RaisingScaler:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("moe backward failed")

    events = []
    original_scaler_cls = omniquant_module.utils.NativeScalerWithGradNormCount
    original_capture_teacher_router_labels = omniquant_module.capture_teacher_router_labels
    original_compute_fp_block_targets = omniquant_module.compute_fp_block_targets
    original_build_moe_label_cache = omniquant_module.build_moe_label_cache
    original_compute_moe_self_supervision_loss = omniquant_module.compute_moe_self_supervision_loss
    original_smooth_and_quant_temporary = omniquant_module.smooth_and_quant_temporary
    original_clear_temp_variable = omniquant_module.clear_temp_variable
    try:
        omniquant_module.utils.NativeScalerWithGradNormCount = lambda *args, **kwargs: RaisingScaler()
        omniquant_module.capture_teacher_router_labels = lambda *args, **kwargs: None
        omniquant_module.compute_fp_block_targets = lambda *args, **kwargs: (
            torch.zeros_like(args[3]),
            torch.zeros_like(args[4]),
        )
        omniquant_module.build_moe_label_cache = lambda *args, **kwargs: [{"expert_labels": {}, "shared_labels": None}]
        omniquant_module.compute_moe_self_supervision_loss = (
            lambda qlayer, quant_inputs, *args, **kwargs: qlayer.mlp.experts.bound_factor * 0
            + torch.tensor(1.0, device=quant_inputs.device)
        )
        omniquant_module.smooth_and_quant_temporary = lambda *args, **kwargs: events.append("smooth")
        omniquant_module.clear_temp_variable = lambda *args, **kwargs: events.append("clear")

        args = SimpleNamespace(
            let=False,
            nsamples=1,
            batch_size=1,
            epochs=1,
            lwc_lr=1e-2,
            linear_lora_lr=1e-4,
            wd=0.0,
            max_grad_norm=None,
            use_linear_lora=False,
            stage0_attn_epochs=0,
            stage1_update_attn=True,
            stage2_enable_calibration=False,
            stage3_enable_block_update=False,
            stage3_aux_loss=False,
            k_loss=1,
            k_routing=1,
            aug_loss=False,
        )

        with pytest.raises(RuntimeError, match="moe backward failed"):
            train_decoupled_moe_layer(
                layer=DummyLayer(),
                qlayer=DummyLayer(),
                args=args,
                logger=DummyLogger(),
                layer_idx=0,
                fp_inps=torch.zeros(1, 1, 1, dtype=torch.float32),
                quant_inps=torch.zeros(1, 1, 1, dtype=torch.float32),
                layer_kwargs={},
                attention_mask_batch=None,
                attention_mask=None,
                position_ids=None,
                traincast=nullcontext,
                use_grad_scaler=False,
                quant_routing_top_n=1,
                use_router_weight_in_loss=False,
            )

        assert events == ["smooth", "clear"]
    finally:
        omniquant_module.utils.NativeScalerWithGradNormCount = original_scaler_cls
        omniquant_module.capture_teacher_router_labels = original_capture_teacher_router_labels
        omniquant_module.compute_fp_block_targets = original_compute_fp_block_targets
        omniquant_module.build_moe_label_cache = original_build_moe_label_cache
        omniquant_module.compute_moe_self_supervision_loss = original_compute_moe_self_supervision_loss
        omniquant_module.smooth_and_quant_temporary = original_smooth_and_quant_temporary
        omniquant_module.clear_temp_variable = original_clear_temp_variable


def test_train_decoupled_moe_layer_tracks_stages_and_releases_label_cache_early():
    class DummyLogger:
        def __init__(self):
            self.infos = []
            self.warnings = []

        def info(self, message, *args, **kwargs):
            self.infos.append(message)

        def warning(self, message, *args, **kwargs):
            self.warnings.append(message)

    class DummyGate(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1, 1))
            self.top_k = 2
            self.out_features = 4

        def forward(self, hidden_states):
            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            logits = torch.ones(flat.shape[0], self.out_features, device=hidden_states.device)
            values, indices = torch.topk(logits, k=self.top_k, dim=-1)
            return logits, values, indices

    class DummyQuantModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.bound_factor = nn.Parameter(torch.tensor(0.0))

        def forward(self, hidden_states, **kwargs):
            return hidden_states

    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = nn.Identity()
            self.post_attention_layernorm = nn.Identity()
            self.self_attn = nn.Module()
            self.self_attn.q_proj = DummyQuantModule()
            self.mlp = nn.Module()
            self.mlp.gate = DummyGate()
            self.mlp.experts = DummyQuantModule()

        def forward(self, hidden_states, **kwargs):
            return hidden_states

    events = []

    class TrackedCache(list):
        def __del__(self):
            events.append("label_cache_released")

    original_build_moe_label_cache = omniquant_module.build_moe_label_cache
    original_compute_moe_self_supervision_loss = omniquant_module.compute_moe_self_supervision_loss
    original_compute_fp_block_targets = omniquant_module.compute_fp_block_targets
    original_capture_teacher_router_labels = omniquant_module.capture_teacher_router_labels
    original_evaluate_block_loss_modes = omniquant_module.evaluate_block_loss_modes
    original_train_expert_shift_calibration_stage = omniquant_module.train_expert_shift_calibration_stage
    original_update_block_parameters_with_loss = omniquant_module.update_block_parameters_with_loss
    original_compute_quantized_expert_shift_metrics = omniquant_module.compute_quantized_expert_shift_metrics
    try:
        def fake_build_moe_label_cache(*args, **kwargs):
            events.append("label_cache_built")
            return TrackedCache([{"expert_labels": {}, "shared_labels": None}])

        def fake_compute_moe_self_supervision_loss(qlayer, quant_inputs, *args, **kwargs):
            return qlayer.mlp.experts.bound_factor * 0 + torch.tensor(1.0, device=quant_inputs.device)

        def fake_compute_fp_block_targets(*args, **kwargs):
            fp_inputs = args[3]
            quant_inputs = args[4]
            return torch.zeros_like(fp_inputs), torch.zeros_like(quant_inputs)

        def fake_capture_teacher_router_labels(*args, **kwargs):
            return (
                torch.zeros(2, 1, 2, dtype=torch.float32),
                torch.zeros(2, 1, 2, dtype=torch.long),
            )

        def fake_evaluate_block_loss_modes(*args, **kwargs):
            events.append(f"block_eval:{kwargs['epoch_idx']}")
            return {"student": {"main": 0.0, "aug": 0.0, "total": 0.0}}

        def fake_train_expert_shift_calibration_stage(*args, **kwargs):
            assert "label_cache_released" in events
            events.append("stage2_train")
            return 0.5

        def fake_update_block_parameters_with_loss(*args, **kwargs):
            events.append("stage3_train")
            return 0.25

        def fake_compute_quantized_expert_shift_metrics(*args, **kwargs):
            events.append(f"shift:{args[4]}")
            return {"any": 0.1, "half": 0.2, "all": 0.3}

        omniquant_module.build_moe_label_cache = fake_build_moe_label_cache
        omniquant_module.compute_moe_self_supervision_loss = fake_compute_moe_self_supervision_loss
        omniquant_module.compute_fp_block_targets = fake_compute_fp_block_targets
        omniquant_module.capture_teacher_router_labels = fake_capture_teacher_router_labels
        omniquant_module.evaluate_block_loss_modes = fake_evaluate_block_loss_modes
        omniquant_module.train_expert_shift_calibration_stage = fake_train_expert_shift_calibration_stage
        omniquant_module.update_block_parameters_with_loss = fake_update_block_parameters_with_loss
        omniquant_module.compute_quantized_expert_shift_metrics = fake_compute_quantized_expert_shift_metrics

        layer = DummyLayer()
        qlayer = DummyLayer()
        args = SimpleNamespace(
            let=False,
            nsamples=2,
            batch_size=1,
            epochs=4,
            lwc_lr=1e-2,
            linear_lora_lr=1e-4,
            wd=0.0,
            max_grad_norm=None,
            use_linear_lora=False,
            stage0_attn_epochs=0,
            stage1_update_attn=True,
            stage2_enable_calibration=True,
            stage2_router_epochs=1,
            stage2_router_lr=1e-3,
            stage2_update_attn=False,
            stage2_use_kl=False,
            stage3_enable_block_update=True,
            stage3_block_update_epochs=1,
            stage3_update_attn=True,
            stage3_update_router=False,
            stage3_update_expert=False,
            stage3_aux_loss=False,
            stage3_aux_loss_weight=0.1,
            block_eval_interval=2,
            k_loss=2,
            k_routing=2,
            train_gate_lora=False,
            train_shared_gate=False,
            calibrate_router=False,
            aug_loss=True,
        )

        fp_inputs = torch.zeros(2, 1, 1, dtype=torch.float32)
        quant_inputs = torch.zeros(2, 1, 1, dtype=torch.float32)
        final_loss, diagnostics = train_decoupled_moe_layer(
            layer=layer,
            qlayer=qlayer,
            args=args,
            logger=DummyLogger(),
            layer_idx=0,
            fp_inps=fp_inputs,
            quant_inps=quant_inputs,
            layer_kwargs={},
            attention_mask_batch=None,
            attention_mask=None,
            position_ids=None,
            traincast=nullcontext,
            use_grad_scaler=False,
            quant_routing_top_n=4,
            use_router_weight_in_loss=False,
        )

        gc.collect()

        assert final_loss == 0.25
        assert diagnostics["expert_shift"]["stage1_pre"]["any"] == 0.1
        assert diagnostics["expert_shift"]["stage1_post"]["half"] == 0.2
        assert diagnostics["expert_shift"]["stage2_post"]["all"] == 0.3
        assert diagnostics["expert_shift"]["stage3_post"]["any"] == 0.1
        assert events.index("label_cache_released") < events.index("stage2_train")
        assert events.count("label_cache_built") == 1
        assert "shift:stage1_pre" in events
        assert "shift:stage1_post" in events
        assert "shift:stage2_post" in events
        assert "shift:stage3_post" in events
        assert "block_eval:1" in events
        assert "block_eval:3" in events
        assert "block_eval:pre_stage2" in events
        assert "block_eval:post_stage2" in events
        assert "block_eval:pre_update" in events
        assert "block_eval:post_update" in events
    finally:
        omniquant_module.build_moe_label_cache = original_build_moe_label_cache
        omniquant_module.compute_moe_self_supervision_loss = original_compute_moe_self_supervision_loss
        omniquant_module.compute_fp_block_targets = original_compute_fp_block_targets
        omniquant_module.capture_teacher_router_labels = original_capture_teacher_router_labels
        omniquant_module.evaluate_block_loss_modes = original_evaluate_block_loss_modes
        omniquant_module.train_expert_shift_calibration_stage = original_train_expert_shift_calibration_stage
        omniquant_module.update_block_parameters_with_loss = original_update_block_parameters_with_loss
        omniquant_module.compute_quantized_expert_shift_metrics = original_compute_quantized_expert_shift_metrics