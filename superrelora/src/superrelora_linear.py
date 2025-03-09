import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SuperReLoRALinear(nn.Module):
    """
    x W + α · (x  W_A W_B)

    partial_merge(alpha_merge):
        W      ← W + α_merge · ΔW
        W_A,B  ← (1-α_merge) · W_A,B
        [scale Adam moments if optimizer.state is provided]
    """

    def __init__(self, in_f, out_f, r=64, alpha=32, dropout=0.0, bias=True):
        super().__init__()
        self.r = r
        self.scale = alpha / r

        # Frozen base weight
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None

        # LoRA pair
        self.lora_A = nn.Linear(in_f, r, bias=False)
        self.lora_B = nn.Linear(r, out_f, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.dropout = nn.Dropout(dropout)

    # ---------- forward ----------
    def forward(self, x: torch.Tensor):
        return (
            F.linear(x, self.weight, self.bias)
            + self.lora_B(self.lora_A(self.dropout(x))) * self.scale
        )

    # ---------- partial merge ----------
    @torch.no_grad()
    def partial_merge(self, alpha=0.1, optimizer_state=None):
        """
        Transfers a portion α to the base weight and decays the LoRA pair.
        """
        delta = (self.lora_B.weight @ self.lora_A.weight) * self.scale
        self.weight.data.add_(alpha * delta)

        self.lora_A.weight.mul_(1 - alpha)
        self.lora_B.weight.mul_(1 - alpha)

        # scale Adam moments
        if optimizer_state is not None:
            for p in (self.lora_A.weight, self.lora_B.weight):
                state = optimizer_state.get(p, {})
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in state:
                        state[key].mul_(1 - alpha)

        return delta.norm().item()

    def unmerge(self):
        """Unmerge LoRA weights from base weights."""
        if not self.merged:
            return
            
        with torch.no_grad():
            # Calculate the unmerge amount
            unmerge_amount = self.merge_ratio * self.scaling
            
            # Restore base weights
            self.weight.data -= unmerge_amount * (self.lora_B @ self.lora_A)
            
            # Reset merge state
            self.merged = False
            self.merge_ratio = 0.0 