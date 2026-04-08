import os
import sys
from contextlib import nullcontext
from types import SimpleNamespace

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
from quantize.block_evaluator import evaluate_block_loss_modes  # noqa: E402
from quantize.omniquant import (  # noqa: E402
    build_block_update_param_groups,
    build_dynamic_moe_label_cache,
    compute_moe_self_supervision_loss,
    forward_layer_with_moe_state,
    get_attention_epochs,
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


def test_get_attention_epochs_defaults_to_global_epochs_and_allows_override():
    assert get_attention_epochs(SimpleNamespace(epochs=12)) == 12
    assert get_attention_epochs(SimpleNamespace(epochs=12, attn_epochs=3)) == 3
    assert get_attention_epochs(SimpleNamespace(epochs=12, attn_epochs=3), layer_idx=0) == 3
    assert get_attention_epochs(SimpleNamespace(epochs=12, attn_epochs=3), layer_idx=1) == 3
    assert get_attention_epochs(SimpleNamespace(epochs=12, attn_epochs=3), layer_idx=2) == 4
    assert get_attention_epochs(SimpleNamespace(epochs=12, attn_epochs=3), layer_idx=5) == 5
    assert get_attention_epochs(SimpleNamespace(epochs=12, attn_epochs=0), layer_idx=6) == 0


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


def test_build_dynamic_moe_label_cache_follows_current_student_selected_experts():
    class IdentityExpert(nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale

        def forward(self, hidden_states):
            return hidden_states * self.scale

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.experts = nn.ModuleList([
                IdentityExpert(1.0),
                IdentityExpert(2.0),
                IdentityExpert(4.0),
            ])
            self.mlp.shared_expert = IdentityExpert(8.0)

    layer = FakeLayer()
    fp_mlp_inputs = torch.tensor([[[1.0], [3.0]]], dtype=torch.float32)
    top_indices = torch.tensor([[[2, 1], [1, 2]]], dtype=torch.long)
    top_weights = torch.tensor([[[0.75, 0.25], [0.6, 0.4]]], dtype=torch.float32)

    label_cache = build_dynamic_moe_label_cache(
        layer,
        fp_mlp_inputs,
        top_indices,
        top_weights,
        use_router_weight_in_loss=True,
    )

    assert set(label_cache[0]["expert_labels"].keys()) == {1, 2}
    assert 0 not in label_cache[0]["expert_labels"]
    expert_two_labels = label_cache[0]["expert_labels"][2]["labels"]
    expert_one_labels = label_cache[0]["expert_labels"][1]["labels"]
    assert torch.allclose(expert_two_labels, torch.tensor([[4.0], [12.0]]))
    assert torch.allclose(expert_one_labels, torch.tensor([[2.0], [6.0]]))
    assert torch.allclose(label_cache[0]["shared_labels"], torch.tensor([[8.0], [24.0]]))


def test_forward_layer_with_moe_state_handles_tensor_and_tuple_layer_outputs():
    class IdentityNorm(nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    class FakeGate(nn.Module):
        def forward(self, hidden_states):
            return torch.softmax(torch.stack([hidden_states[:, 0], hidden_states[:, 1]], dim=-1), dim=-1)

    class FakeMlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = FakeGate()

        def forward(self, hidden_states):
            _ = self.gate(hidden_states.reshape(-1, hidden_states.shape[-1]))
            return hidden_states + 1.0

    class TensorReturnLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = IdentityNorm()
            self.post_attention_layernorm = IdentityNorm()
            self.self_attn = nn.Identity()
            self.mlp = FakeMlp()

        def forward(self, hidden_states, **_kwargs):
            return self.mlp(hidden_states)

    class TupleReturnLayer(TensorReturnLayer):
        def forward(self, hidden_states, output_router_logits=False, **_kwargs):
            outputs = (self.mlp(hidden_states),)
            if output_router_logits:
                outputs += (torch.ones(hidden_states.shape[0], hidden_states.shape[1], 2),)
            return outputs

    hidden_states = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)

    tensor_out, tensor_mlp_inputs, tensor_router_scores = forward_layer_with_moe_state(
        TensorReturnLayer(),
        hidden_states,
        layer_kwargs={},
    )
    tuple_out, tuple_mlp_inputs, tuple_router_scores = forward_layer_with_moe_state(
        TupleReturnLayer(),
        hidden_states,
        layer_kwargs={"output_router_logits": True},
    )

    assert torch.allclose(tensor_out, hidden_states + 1.0)
    assert torch.allclose(tuple_out, hidden_states + 1.0)
    assert torch.allclose(tensor_mlp_inputs, hidden_states)
    assert torch.allclose(tuple_mlp_inputs, hidden_states)
    assert tensor_router_scores.shape == (2, 2)
    assert tuple_router_scores.shape == (2, 2)


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
        block_loss_attn=False,
        block_loss_router=False,
        block_loss_expert=False,
        expert_loss_expert=False,
    )

    selected_params, param_groups = build_block_update_param_groups(qlayer, args)
    assert selected_params == []
    assert param_groups == []

    args.block_loss_attn = True
    selected_params, param_groups = build_block_update_param_groups(qlayer, args)
    selected_ids = {id(param) for param in selected_params}
    assert id(qlayer.self_attn.q_proj.bound_factor) in selected_ids
    assert id(qlayer.self_attn.q_proj.lora_A) in selected_ids
    assert id(qlayer.self_attn.q_proj.lora_B) in selected_ids
    assert id(qlayer.mlp.gate.weight) not in selected_ids
    assert len(param_groups) == 2

    args.block_loss_attn = False
    args.block_loss_router = True
    selected_params, param_groups = build_block_update_param_groups(qlayer, args)
    selected_ids = {id(param) for param in selected_params}
    assert id(qlayer.self_attn.q_proj.bound_factor) not in selected_ids
    assert id(qlayer.mlp.gate.weight) in selected_ids
    assert id(qlayer.mlp.shared_expert_gate.weight) in selected_ids
    assert len(param_groups) == 2


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