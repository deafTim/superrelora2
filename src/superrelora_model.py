from typing import Any, List, Optional

import torch
import torch.nn as nn

from src.superrelora_linear import SuperReLoRALinear


class SuperReLoRaModel(nn.Module):
    """
    Wrapper over AutoModelForCausalLM.
    Replaces target nn.Linear layers with SuperReLoRALinear.

    orthogonal_reinit=True  -> SuperReLoRa (column-space orthogonalize at reinit)
    orthogonal_reinit=False -> ReLoRA baseline (merge + random reinit only)
    reinit_momentum μ       -> partial merge + blend old/new A,B (smooth cycle transition)
    """

    def __init__(
        self,
        base_model: nn.Module,
        *,
        r: int = 64,
        alpha: int = 32,
        target_modules: Optional[List[str]] = None,
        orthogonal_reinit: bool = True,
        prune_ratio: float = 0.99,
        reinit_momentum: float = 0.0,
    ):
        super().__init__()
        self.model = base_model
        self.r = r
        self.alpha = alpha
        self.target_modules = target_modules or []
        self.orthogonal_reinit = orthogonal_reinit
        self.prune_ratio = prune_ratio
        self.reinit_momentum = float(reinit_momentum)

        self._patch_linear_layers()

    def _patch_linear_layers(self) -> None:
        self.replaced_modules: list[str] = []

        for name, module in list(self.model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if self.target_modules and not any(key in name for key in self.target_modules):
                continue

            parent = self._get_parent(name)
            child_name = name.split(".")[-1]
            old_linear: nn.Linear = getattr(parent, child_name)

            if isinstance(old_linear, SuperReLoRALinear):
                continue

            new_linear = SuperReLoRALinear(
                in_f=old_linear.in_features,
                out_f=old_linear.out_features,
                r=self.r,
                alpha=self.alpha,
                dropout=0.0,
                bias=old_linear.bias is not None,
            )
            # Match base Linear dtype/device (critical for fp16/bf16 amp).
            new_linear = new_linear.to(
                device=old_linear.weight.device,
                dtype=old_linear.weight.dtype,
            )
            new_linear.weight.data.copy_(old_linear.weight.data)
            new_linear.weight.requires_grad_(False)
            if old_linear.bias is not None:
                new_linear.bias.data.copy_(old_linear.bias.data)
                new_linear.bias.requires_grad_(False)
            # U buffer stays float32 for QR numerics
            if new_linear.U is not None:
                new_linear.U = new_linear.U.float()

            setattr(parent, child_name, new_linear)
            self.replaced_modules.append(name)

    def _get_parent(self, module_name: str) -> nn.Module:
        parts = module_name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def iter_lora_layers(self):
        for module in self.model.modules():
            if isinstance(module, SuperReLoRALinear):
                yield module

    @torch.no_grad()
    def step_merge_reinit(
        self,
        step: int,
        every: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> float:
        """Merge + reinit (+ optional orthogonalize) every `every` steps."""
        if every is None or every <= 0 or step <= 0 or step % every:
            return 0.0

        total_norm = 0.0
        opt_state = optimizer.state if optimizer is not None else None
        for module in self.iter_lora_layers():
            total_norm += module.merge_and_reinit(
                optimizer_state=opt_state,
                orthogonal=self.orthogonal_reinit,
                prune_ratio=self.prune_ratio,
                reinit_momentum=self.reinit_momentum,
            )
        return total_norm

    # Backwards-compatible alias
    def step_partial_merge(self, step: int, every: int, merge_alpha: float = 1.0, optimizer=None):
        return self.step_merge_reinit(step=step, every=every, optimizer=optimizer)
