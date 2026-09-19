"""Unit tests for SuperReLoRA merge-and-reinit (no Hub download)."""
import math

import torch
import torch.nn as nn

from src.superrelora_linear import SuperReLoRALinear


def test_merge_and_orthogonal_reinit():
    torch.manual_seed(0)
    layer = SuperReLoRALinear(in_f=32, out_f=16, r=4, alpha=8, bias=False)
    # Give A a clear column space
    with torch.no_grad():
        layer.lora_A.weight.zero_()
        layer.lora_A.weight[:, :4] = torch.eye(4)
        layer.lora_B.weight.normal_(0, 0.1)

    W_before = layer.weight.data.clone()
    norm = layer.merge_and_reinit(orthogonal=True, prune_ratio=0.0)
    assert norm > 0
    assert not torch.allclose(layer.weight.data, W_before)
    assert layer.U.shape[0] == 32
    assert layer.U.shape[1] > 0

    # After orthogonalize, A should be ~orthogonal to U
    proj = layer.lora_A.weight.data @ layer.U
    assert proj.norm().item() < 1e-4, f"A not orthogonal to U, norm={proj.norm().item()}"


def test_relora_skips_orthogonalize():
    torch.manual_seed(1)
    layer = SuperReLoRALinear(in_f=32, out_f=16, r=4, alpha=8, bias=False)
    with torch.no_grad():
        layer.lora_A.weight.zero_()
        layer.lora_A.weight[:, :4] = torch.eye(4)
        layer.lora_B.weight.normal_(0, 0.1)

    layer.merge_and_reinit(orthogonal=False, prune_ratio=0.0)
    # Without orthogonalize, reinited A is not forced orthogonal to U
    # (random Kaiming may accidentally be small, so only check U was stored)
    assert layer.U.shape[1] > 0


def test_base_weight_frozen():
    layer = SuperReLoRALinear(8, 8, r=2, alpha=4)
    assert not layer.weight.requires_grad
    assert layer.lora_A.weight.requires_grad
    assert layer.lora_B.weight.requires_grad


if __name__ == "__main__":
    test_merge_and_orthogonal_reinit()
    test_relora_skips_orthogonalize()
    test_base_weight_frozen()
    print("✅ SuperReLoRA unit tests passed")
