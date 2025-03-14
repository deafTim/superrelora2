from transformers import AutoModelForCausalLM
from src.superrelora_model import SuperReLoRaModel
import torch
import os

model_id = "nicholasKluge/TeenyTinyLlama-160m"
cache_dir = "models/teenytinyllama-160m"

# Always use HuggingFace Hub and cache locally
base = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    cache_dir=cache_dir
)

model = SuperReLoRaModel(base, r=16, alpha=8, target_modules=["q_proj", "k_proj", "v_proj"])

input_ids = torch.randint(0, base.config.vocab_size, (1, 16))
with torch.no_grad():
    out = model(input_ids=input_ids)
print("logits:", out.logits.shape)

model.step_partial_merge(step=500, every=500, merge_alpha=0.1)
print("partial merge OK") 