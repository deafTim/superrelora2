"""Unit tests for SuperReLoRA merge-and-reinit (no Hub download)."""

import torch

from src.superrelora_linear import SuperReLoRALinear


def test_merge_and_orthogonal_reinit():
    torch.manual_seed(0)
    layer = SuperReLoRALinear(in_f=32, out_f=16, r=4, alpha=8, bias=False)
    with torch.no_grad():
        layer.lora_A.weight.zero_()
        layer.lora_A.weight[:, :4] = torch.eye(4)
        layer.lora_B.weight.normal_(0, 0.1)

    W_before = layer.weight.data.clone()
    norm = layer.merge_and_reinit(orthogonal=True, prune_ratio=0.0, reinit_momentum=0.0)
    assert norm > 0
    assert not torch.allclose(layer.weight.data, W_before)
    assert layer.U.shape[0] == 32
    assert layer.U.shape[1] > 0

    proj = layer.lora_A.weight.data @ layer.U
    assert proj.norm().item() < 1e-4, f"A not orthogonal to U, norm={proj.norm().item()}"


def test_relora_skips_orthogonalize():
    torch.manual_seed(1)
    layer = SuperReLoRALinear(in_f=32, out_f=16, r=4, alpha=8, bias=False)
    with torch.no_grad():
        layer.lora_A.weight.zero_()
        layer.lora_A.weight[:, :4] = torch.eye(4)
        layer.lora_B.weight.normal_(0, 0.1)

    layer.merge_and_reinit(orthogonal=False, prune_ratio=0.0, reinit_momentum=0.0)
    assert layer.U.shape[1] > 0


def test_reinit_momentum_partial_merge_and_blend():
    torch.manual_seed(2)
    layer = SuperReLoRALinear(in_f=32, out_f=16, r=4, alpha=8, bias=False)
    with torch.no_grad():
        layer.lora_A.weight.normal_(0, 0.5)
        layer.lora_B.weight.normal_(0, 0.5)
    A_old = layer.lora_A.weight.data.clone()
    B_old = layer.lora_B.weight.data.clone()
    W_before = layer.weight.data.clone()
    delta = (B_old.float() @ A_old.float()) * layer.scale
    mu = 0.25

    layer.merge_and_reinit(orthogonal=True, prune_ratio=0.0, reinit_momentum=mu)

    expected_W = W_before + (1.0 - mu) * delta.to(dtype=W_before.dtype)
    assert torch.allclose(layer.weight.data, expected_W, atol=1e-5)
    assert torch.allclose(layer.lora_B.weight.data, mu * B_old, atol=1e-5)
    cos = torch.nn.functional.cosine_similarity(
        layer.lora_A.weight.data.flatten().float(),
        A_old.flatten().float(),
        dim=0,
    )
    assert cos.item() > 0.05


def test_base_weight_frozen():
    layer = SuperReLoRALinear(8, 8, r=2, alpha=4)
    assert not layer.weight.requires_grad
    assert layer.lora_A.weight.requires_grad
    assert layer.lora_B.weight.requires_grad


if __name__ == "__main__":
    test_merge_and_orthogonal_reinit()
    test_relora_skips_orthogonalize()
    test_reinit_momentum_partial_merge_and_blend()
    test_base_weight_frozen()
    print("✅ SuperReLoRA unit tests passed")
