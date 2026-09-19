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


class SuperReLoRALinear(nn.Module):
    """
    Frozen base weight + LoRA:
        y = x W^T + scale * B(A(x))

    merge_and_reinit:
        W <- W + scale * (B @ A)
        update orthonormal basis U of past A column-spaces (input directions)
        reinit A, B
        if orthogonal: A <- A (I - U U^T)
        prune Adam moments for A, B
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
        return (
            F.linear(x, self.weight, self.bias)
            + self.lora_B(self.lora_A(self.dropout(x))) * self.scale
        )

    @torch.no_grad()
    def _append_basis_from_A(self) -> None:
        """Accumulate column-space of current A into U.

        lora_A.weight is (r, in_f); paper W_A is (in_f, r), so columns are A.T.
        """
        A_cols = self.lora_A.weight.data.T.contiguous()  # (in_f, r)
        # Drop near-zero columns for numerical stability
        col_norms = A_cols.norm(dim=0)
        keep = col_norms > 1e-8
        if not keep.any():
            return
        A_cols = A_cols[:, keep]
        Q = torch.linalg.qr(A_cols, mode="reduced").Q
        if self.U.numel() == 0:
            self.U = Q
        else:
            stacked = torch.cat([self.U, Q], dim=1)
            self.U = torch.linalg.qr(stacked, mode="reduced").Q

    @torch.no_grad()
    def _orthogonalize_A(self) -> None:
        """Project A onto the orthogonal complement of span(U): A <- A (I - UU^T)."""
        if self.U.numel() == 0:
            return
        # (r, in) - ((r, in) @ (in, k)) @ (k, in)
        self.lora_A.weight.data.sub_(
            (self.lora_A.weight.data @ self.U) @ self.U.T
        )

    @torch.no_grad()
    def merge_and_reinit(
        self,
        optimizer_state: Optional[dict] = None,
        orthogonal: bool = True,
        prune_ratio: float = 0.99,
    ) -> float:
        """Full merge, optional column-space orthogonal reinit, Adam prune."""
        delta = (self.lora_B.weight @ self.lora_A.weight) * self.scale
        delta_norm = delta.norm().item()
        self.weight.data.add_(delta)

        self._append_basis_from_A()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        if orthogonal:
            self._orthogonalize_A()

        if optimizer_state is not None:
            for p in (self.lora_A.weight, self.lora_B.weight):
                state = optimizer_state.get(p, None)
                if state is not None:
                    prune_adam_state(state, prune_ratio=prune_ratio)

        return delta_norm
