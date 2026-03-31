import torch
import torch.nn as nn
import torch.nn.functional as F

from quantize.int_linear import QuantLinear


def _build_linear_from_weight(weight: torch.Tensor) -> nn.Linear:
    linear = nn.Linear(
        weight.shape[1],
        weight.shape[0],
        bias=False,
        device=weight.device,
        dtype=weight.dtype,
    )
    with torch.no_grad():
        linear.weight.copy_(weight)
    return linear


class QuantExpertMLP(nn.Module):
    def __init__(
        self,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        act_fn,
        args,
    ):
        super().__init__()
        self.gate_proj = QuantLinear(
            _build_linear_from_weight(gate_weight),
            args.weight_quant_params,
            args.act_quant_params,
            use_linear_lora=getattr(args, "use_linear_lora", False),
            linear_lora_r=getattr(args, "linear_lora_r", 16),
            linear_lora_alpha=getattr(args, "linear_lora_alpha", 16.0),
        )
        self.up_proj = QuantLinear(
            _build_linear_from_weight(up_weight),
            args.weight_quant_params,
            args.act_quant_params,
            use_linear_lora=getattr(args, "use_linear_lora", False),
            linear_lora_r=getattr(args, "linear_lora_r", 16),
            linear_lora_alpha=getattr(args, "linear_lora_alpha", 16.0),
        )
        self.down_proj = QuantLinear(
            _build_linear_from_weight(down_weight),
            args.weight_quant_params,
            args.act_quant_params,
            use_linear_lora=getattr(args, "use_linear_lora", False),
            linear_lora_r=getattr(args, "linear_lora_r", 16),
            linear_lora_alpha=getattr(args, "linear_lora_alpha", 16.0),
        )
        self.act_fn = act_fn

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gated = self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.down_proj(gated)


class QuantizedPackedExperts(nn.Module):
    def __init__(self, org_module: nn.Module, args):
        super().__init__()
        self.num_experts = org_module.num_experts
        self.hidden_dim = org_module.hidden_dim
        self.intermediate_dim = org_module.intermediate_dim
        self.act_fn = org_module.act_fn
        self.experts = nn.ModuleList()

        for expert_idx in range(self.num_experts):
            gate_weight, up_weight = org_module.gate_up_proj[expert_idx].chunk(2, dim=0)
            down_weight = org_module.down_proj[expert_idx]
            self.experts.append(
                QuantExpertMLP(
                    gate_weight=gate_weight.detach().clone(),
                    up_weight=up_weight.detach().clone(),
                    down_weight=down_weight.detach().clone(),
                    act_fn=self.act_fn,
                    args=args,
                )
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_id = int(expert_idx.item())
            top_k_pos, token_idx = torch.where(expert_mask[expert_id])
            current_state = hidden_states[token_idx]
            current_hidden_states = self.experts[expert_id](current_state)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


def is_packed_experts_module(module: nn.Module) -> bool:
    return (
        isinstance(module, nn.Module)
        and hasattr(module, "gate_up_proj")
        and hasattr(module, "down_proj")
        and isinstance(getattr(module, "gate_up_proj"), nn.Parameter)
        and isinstance(getattr(module, "down_proj"), nn.Parameter)
        and hasattr(module, "num_experts")
        and hasattr(module, "act_fn")
    )


def get_moe_top_k(moe_module: nn.Module) -> int | None:
    return getattr(moe_module, "top_k", None)


def get_shared_expert_module(moe_module: nn.Module) -> nn.Module | None:
    if hasattr(moe_module, "shared_expert"):
        return moe_module.shared_expert
    if hasattr(moe_module, "shared_experts"):
        return moe_module.shared_experts
    return None


def get_moe_experts_module(moe_module: nn.Module) -> nn.Module | None:
    return getattr(moe_module, "experts", None)


def compute_router_scores(moe_module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    gate_output = moe_module.gate(flat_hidden_states)

    if isinstance(gate_output, tuple):
        return gate_output[0].float()

    if isinstance(moe_module.gate, nn.Linear):
        return torch.softmax(gate_output.float(), dim=-1)

    return torch.sigmoid(gate_output.float())


def select_top_n_experts(router_scores: torch.Tensor, top_n: int) -> tuple[torch.Tensor, torch.Tensor]:
    top_n = min(top_n, router_scores.shape[-1])
    top_weights, top_indices = torch.topk(router_scores, k=top_n, dim=-1)
    top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return top_indices, top_weights


def compute_expert_down_proj_output(
    experts_module: nn.Module,
    expert_idx: int,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    if hasattr(experts_module, "experts") and isinstance(experts_module.experts, nn.ModuleList):
        return experts_module.experts[expert_idx](hidden_states)

    gate_up_weight = experts_module.gate_up_proj[expert_idx]
    down_proj_weight = experts_module.down_proj[expert_idx]
    gate, up = F.linear(hidden_states, gate_up_weight).chunk(2, dim=-1)
    current_hidden_states = experts_module.act_fn(gate) * up
    return F.linear(current_hidden_states, down_proj_weight)


def pin_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    cpu_tensor = tensor.detach().to(device="cpu")
    if torch.cuda.is_available():
        return cpu_tensor.pin_memory()
    return cpu_tensor