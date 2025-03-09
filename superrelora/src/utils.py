import torch
from torch.optim import Optimizer
from typing import Dict, Any, Optional

def scale_optimizer_state(optimizer: Optimizer, scale: float):
    """Scale the optimizer state by a given factor."""
    for param_group in optimizer.param_groups:
        for p in param_group['params']:
            if p in optimizer.state:
                state = optimizer.state[p]
                for key in state:
                    if isinstance(state[key], torch.Tensor):
                        state[key] = state[key] * scale

def get_trainable_params(model: torch.nn.Module) -> int:
    """Get the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size(model: torch.nn.Module) -> int:
    """Get the total number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())

def count_superrelora_params(model: torch.nn.Module) -> Dict[str, int]:
    """Count the number of parameters in SuperReLoRA layers."""
    total_params = 0
    lora_params = 0
    
    for module in model.modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            total_params += module.weight.numel() + module.bias.numel()
            lora_params += module.lora_A.numel() + module.lora_B.numel()
    
    return {
        'total_params': total_params,
        'lora_params': lora_params,
        'base_params': total_params - lora_params
    }

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optimizer,
    epoch: int,
    path: str,
    extra_state: Optional[Dict[str, Any]] = None
):
    """Save model checkpoint with optimizer state."""
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    if extra_state:
        state.update(extra_state)
    torch.save(state, path)

def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optimizer,
    path: str
) -> Dict[str, Any]:
    """Load model checkpoint with optimizer state."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return {k: v for k, v in checkpoint.items() if k not in ['model_state_dict', 'optimizer_state_dict']} 