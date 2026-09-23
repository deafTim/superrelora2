import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def prune_adam_state(state: dict, prune_ratio: float = 0.99) -> None:
    """Zero out the lowest-magnitude fraction of Adam moments (ReLoRA-style)."""
    if prune_ratio <= 0:
        return
    for key in ("exp_avg", "exp_avg_sq"):
        if key not in state:
            continue
        tensor = state[key]
        if tensor is None or tensor.numel() == 0:
            continue
        flat = tensor.detach().abs().reshape(-1)
        k = int(prune_ratio * flat.numel())
        if k <= 0:
            continue
        if k >= flat.numel():
            tensor.zero_()
            continue
        threshold = torch.kthvalue(flat, k).values
        tensor.mul_(tensor.abs() >= threshold)


def decay_adam_state(state: dict, keep: float) -> None:
    """Scale Adam moments by `keep` to carry optimizer momentum across reinit."""
    if keep <= 0:
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state and state[key] is not None:
                state[key].zero_()
        return
    if keep >= 1:
        return
    for key in ("exp_avg", "exp_avg_sq"):
        if key in state and state[key] is not None:
            state[key].mul_(keep)


class SuperReLoRALinear(nn.Module):
    """
    Frozen base weight + LoRA:
        y = x W^T + scale * B(A(x))

    merge_and_reinit (reinit_momentum μ ∈ [0, 1]):
        W <- W + (1-μ) * scale * (B @ A)     # partial merge
        update orthonormal basis U from current A
        reinit A, B; if orthogonal: A <- A (I - U U^T)
        A <- (1-μ) A_new + μ A_old           # smooth unfinished cycle
        B <- (1-μ) B_new + μ B_old
        prune / decay Adam moments for A, B
    """

    def __init__(self, in_f, out_f, r=64, alpha=32, dropout=0.0, bias=True):
        super().__init__()
        self.r = r
        self.in_f = in_f
        self.out_f = out_f
        self.scale = alpha / r

        self.weight = nn.Parameter(torch.empty(out_f, in_f), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_f), requires_grad=False) if bias else None

        self.lora_A = nn.Linear(in_f, r, bias=False)
        self.lora_B = nn.Linear(r, out_f, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.dropout = nn.Dropout(dropout)
        # Orthonormal basis of past adapter input subspaces, shape (in_f, n_dirs)
        self.register_buffer("U", torch.zeros(in_f, 0), persistent=True)

    def forward(self, x: torch.Tensor):
        # Keep compute dtype in sync with activations (fp16/bf16 amp).
        weight = self.weight.to(dtype=x.dtype)
        bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
        lora_x = self.dropout(x)
        a_w = self.lora_A.weight.to(dtype=x.dtype)
        b_w = self.lora_B.weight.to(dtype=x.dtype)
        lora_out = F.linear(F.linear(lora_x, a_w), b_w)
        return F.linear(x, weight, bias) + lora_out * self.scale

    @torch.no_grad()
    def _append_basis_from_A(self) -> None:
        """Accumulate column-space of current A into U.

        lora_A.weight is (r, in_f); paper W_A is (in_f, r), so columns are A.T.
        QR is done in float32 for stability.
        """
        A_cols = self.lora_A.weight.data.detach().float().T.contiguous()  # (in_f, r)
        col_norms = A_cols.norm(dim=0)
        keep = col_norms > 1e-8
        if not keep.any():
            return
        A_cols = A_cols[:, keep]
        # QR .Q can be non-contiguous; safetensors requires contiguous buffers.
        Q = torch.linalg.qr(A_cols, mode="reduced").Q.contiguous()
        if self.U.numel() == 0:
            self.U = Q
        else:
            stacked = torch.cat([self.U.float(), Q], dim=1)
            self.U = torch.linalg.qr(stacked, mode="reduced").Q.contiguous()

    @torch.no_grad()
    def _orthogonalize_A(self) -> None:
        """Project A onto the orthogonal complement of span(U): A <- A (I - UU^T)."""
        if self.U.numel() == 0:
            return
        A = self.lora_A.weight.data
        A_f = A.float()
        U = self.U.float()
        A.copy_((A_f - (A_f @ U) @ U.T).to(dtype=A.dtype))

    @torch.no_grad()
    def merge_and_reinit(
        self,
        optimizer_state: Optional[dict] = None,
        orthogonal: bool = True,
        prune_ratio: float = 0.99,
        reinit_momentum: float = 0.0,
    ) -> float:
        """Partial merge + reinit with optional orthogonalization and transition momentum.

        reinit_momentum μ ∈ [0, 1]:
          - μ = 0: full merge into W, hard A/B reset (classic ReLoRA / SuperReLoRa).
          - μ > 0: merge only (1-μ) of BA into W; blend new A/B with old adapters so an
            unfinished cycle is not discarded; Adam moments are scaled by μ after prune.
        """
        mu = float(reinit_momentum)
        mu = 0.0 if mu < 0 else (1.0 if mu > 1 else mu)

        dtype = self.weight.dtype
        A_old = self.lora_A.weight.data.clone()
        B_old = self.lora_B.weight.data.clone()

        delta = (B_old.float() @ A_old.float()) * self.scale
        delta_norm = delta.norm().item()
        # Absorb only (1-μ); residual stays in the blended adapters below.
        self.weight.data.add_(((1.0 - mu) * delta).to(dtype=dtype))

        self._append_basis_from_A()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.lora_A.weight.data = self.lora_A.weight.data.to(dtype=dtype)
        self.lora_B.weight.data = self.lora_B.weight.data.to(dtype=dtype)

        if orthogonal:
            self._orthogonalize_A()

        if mu > 0:
            # Soft transition: keep a fraction of the previous (possibly unfinished) cycle.
            self.lora_A.weight.data.mul_(1.0 - mu).add_(A_old.to(dtype=dtype), alpha=mu)
            self.lora_B.weight.data.mul_(1.0 - mu).add_(B_old.to(dtype=dtype), alpha=mu)

        if optimizer_state is not None:
            for p in (self.lora_A.weight, self.lora_B.weight):
                state = optimizer_state.get(p, None)
                if state is None:
                    continue
                prune_adam_state(state, prune_ratio=prune_ratio)
                if mu > 0:
                    decay_adam_state(state, keep=mu)

        return delta_norm
