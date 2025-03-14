import torch
import torch.nn as nn
from typing import List
from .superrelora_linear import SuperReLoRALinear


class SuperReLoRaModel(nn.Module):
    """
    Wrapper: replaces Linear layers in base_model with SuperReLoRALinear.
    """

    def __init__(
        self,
        base_model: nn.Module,
        *,
        r: int = 64,
        alpha: int = 32,
        target_modules: List[str] = None,
    ):
        super().__init__()
        self.model = base_model
        self.r = r
        self.alpha = alpha
        self.target_modules = target_modules or []

        self._patch_linear_layers()

    # ---------- replace Linear with SuperReLoRALinear ----------
    def _patch_linear_layers(self):
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if self.target_modules and not any(key in name for key in self.target_modules):
                continue

            parent = self._get_parent(name)
            child_name = name.split(".")[-1]
            old_linear: nn.Linear = getattr(parent, child_name)

            new_linear = SuperReLoRALinear(
                in_f=old_linear.in_features,
                out_f=old_linear.out_features,
                r=self.r,
                alpha=self.alpha,
                dropout=0.0,
                bias=old_linear.bias is not None,
            )
            new_linear.weight.data.copy_(old_linear.weight.data)
            if old_linear.bias is not None:
                new_linear.bias.data.copy_(old_linear.bias.data)

            setattr(parent, child_name, new_linear)

    # helper to find parent module
    def _get_parent(self, module_name: str):
        parts = module_name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent

    # ---------- API ----------
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def step_partial_merge(self, step: int, every: int, merge_alpha: float, optimizer=None):
        if every <= 0 or (step % every):
            return 0.0

        total_norm = 0.0
        opt_state = optimizer.state if optimizer is not None else None
        for module in self.model.modules():
            if isinstance(module, SuperReLoRALinear):
                total_norm += module.partial_merge(merge_alpha, opt_state)
        return total_norm 