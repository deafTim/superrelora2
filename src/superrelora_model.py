import torch
import torch.nn as nn
from typing import List, Any

from src.superrelora_linear import SuperReLoRALinear


class SuperReLoRaModel(nn.Module):
    """
    Wrapper over the base model (AutoModelForCausalLM).
    Replaces all nn.Linear (or specified target_modules) with SuperReLoRALinear
    and transparently proxies other methods/attributes of the HuggingFace model
    (generate, save_pretrained, config, etc.).
    """

    def __init__(
        self,
        base_model: nn.Module,
        *,
        r: int = 64,
        alpha: int = 32,
        target_modules: List[str] | None = None,
    ):
        super().__init__()
        self.model = base_model
        self.r = r
        self.alpha = alpha
        self.target_modules = target_modules or []

        # Replace Linear layers with SuperReLoRA equivalents
        self._patch_linear_layers()

    # --------------------------------------------------------------------- #
    #                           INTERNAL HELPERS                             #
    # --------------------------------------------------------------------- #
    def _patch_linear_layers(self) -> None:
        """Recursively traverses the entire model and replaces the required nn.Linear layers."""
        self.replaced_modules: list[str] = []

        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if self.target_modules and not any(key in name for key in self.target_modules):
                continue

            parent = self._get_parent(name)
            child_name = name.split(".")[-1]
            old_linear: nn.Linear = getattr(parent, child_name)

            # if already replaced - skip
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
            # copy weights / bias
            new_linear.weight.data.copy_(old_linear.weight.data)
            if old_linear.bias is not None:
                new_linear.bias.data.copy_(old_linear.bias.data)

            setattr(parent, child_name, new_linear)
            self.replaced_modules.append(name)

    def _get_parent(self, module_name: str) -> nn.Module:
        """Returns the parent nn.Module by its full dotted path."""
        parts = module_name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent

    # --------------------------------------------------------------------- #
    #                             PUBLIC  API                                #
    # --------------------------------------------------------------------- #
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # proxy generate directly to the "inner" model
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    # universal proxy for other methods/attributes
    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    # ------------------- SuperReLoRA-specific methods ---------------- #
    @torch.no_grad()
    def step_partial_merge(
        self,
        step: int,
        every: int,
        merge_alpha: float,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> float:
        """Performs partial merge (α-merge) every `every` steps."""
        if every <= 0 or step % every:
            return 0.0

        total_norm = 0.0
        opt_state = optimizer.state if optimizer is not None else None
        for module in self.model.modules():
            if isinstance(module, SuperReLoRALinear):
                total_norm += module.partial_merge(merge_alpha, opt_state)
        return total_norm

    @torch.no_grad()
    def merge_all(self, alpha: float = 1.0, optimizer=None) -> float:
        """Forcefully merges all LoRA diffs with coefficient alpha."""
        total_norm = 0.0
        opt_state = optimizer.state if optimizer is not None else None
        for module in self.model.modules():
            if isinstance(module, SuperReLoRALinear):
                total_norm += module.partial_merge(alpha, opt_state)
        return total_norm

    @torch.no_grad()
    def unmerge_all(self) -> None:
        """Cancels any previous merge, returning to pure LoRA mode."""
        for module in self.model.modules():
            if isinstance(module, SuperReLoRALinear):
                module.unmerge()
