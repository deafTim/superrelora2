import json
import math
import os
import argparse

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.superrelora_model import SuperReLoRaModel


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate SuperReLoRA / ReLoRA model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint or final_model dir')
    parser.add_argument('--dataset_name', type=str, default='Salesforce/wikitext', help='Dataset name')
    parser.add_argument('--dataset_config', type=str, default='wikitext-2-raw-v1', help='Dataset config')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--max_length', type=int, default=128, help='Max sequence length')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to evaluate')
    parser.add_argument(
        '--method',
        type=str,
        default='superrelora',
        choices=['superrelora', 'relora', 'lora'],
        help='Must match training method (orthogonal_reinit on/off)',
    )
    parser.add_argument(
        '--metrics_out',
        type=str,
        default=None,
        help='Optional path to write metrics.json',
    )
    return parser.parse_args()


def resolve_weight_file(model_path: str) -> str:
    """Accept a .bin/.pt/.safetensors file or a Trainer save directory."""
    if os.path.isfile(model_path):
        return model_path
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"model_path not found: {model_path}")
    for name in (
        "model.safetensors",
        "pytorch_model.bin",
        "model.bin",
        "pytorch_model.pt",
    ):
        candidate = os.path.join(model_path, name)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"No weight file in {model_path} (expected model.safetensors or pytorch_model.bin)"
    )


def load_state_dict(path: str, map_location):
    path = resolve_weight_file(path)
    print(f"Loading weights from: {path}")
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path, device="cpu")
    checkpoint = torch.load(path, map_location=map_location)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint

def compute_perplexity(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0
    num_batches = len(dataloader)
    
    print(f"\nStarting evaluation on {num_batches} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing metrics")):
            # Convert lists to tensors if needed
            if isinstance(batch['input_ids'], list):
                # Если это список тензоров, склеиваем их
                input_ids = torch.stack([x if torch.is_tensor(x) else torch.tensor(x) for x in batch['input_ids']]).to(device)
            elif not torch.is_tensor(batch['input_ids']):
                input_ids = torch.tensor(batch['input_ids']).to(device)
            else:
                input_ids = batch['input_ids'].to(device)
            
            if isinstance(batch['attention_mask'], list):
                attention_mask = torch.stack([x if torch.is_tensor(x) else torch.tensor(x) for x in batch['attention_mask']]).to(device)
            elif not torch.is_tensor(batch['attention_mask']):
                attention_mask = torch.tensor(batch['attention_mask']).to(device)
            else:
                attention_mask = batch['attention_mask'].to(device)
            
            print(f"\nProcessing batch {batch_idx + 1}/{num_batches}")
            print(f"Batch shape: {input_ids.shape}")
            
            # Ensure [batch, seq]
            if input_ids.shape[0] < input_ids.shape[1]:
                input_ids = input_ids.T
                attention_mask = attention_mask.T
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            loss = outputs.loss
            
            mask = attention_mask[:, 1:].contiguous()
            # число предсказанных (не-паддинг) токенов в пачке
            token_count = mask.sum().item()
            # накапливаем суммарный кросс-энтропийный лосс по токенам
            total_loss += loss.item() * token_count  
            total_tokens += token_count

            # Compute accuracy
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            # Shift predictions and labels for next token prediction
            predictions = predictions[:, :-1].contiguous()
            labels = input_ids[:, 1:].contiguous()

            

            mask = attention_mask[:, 1:].contiguous()
            correct = (predictions == labels) * mask
            total_correct += correct.sum().item()

            # Print batch metrics
            batch_loss = loss.item()
            batch_accuracy = correct.sum().item() / mask.sum().item()
            print(f"Batch {batch_idx + 1} - Loss: {batch_loss:.4f}, Accuracy: {batch_accuracy:.4f}")
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    accuracy = total_correct / total_tokens
    
    print(f"\nEvaluation complete!")
    print(f"Total tokens processed: {total_tokens}")
    print(f"Total correct predictions: {total_correct}")
    
    return avg_loss, perplexity, accuracy

def generate_text(model, tokenizer, prompt, max_length=100):
    model.eval()
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prefer GPFS cache if sbatch exported HF_HOME / SUPERRELORA_HF_HOME
    hf_home = os.environ.get("SUPERRELORA_HF_HOME") or os.environ.get("HF_HOME")
    hub_kw = {}
    ds_kw = {}
    if hf_home:
        hub = os.path.join(hf_home, "hub")
        ds = os.path.join(hf_home, "datasets")
        os.makedirs(hub, exist_ok=True)
        os.makedirs(ds, exist_ok=True)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hub)
        os.environ.setdefault("HF_DATASETS_CACHE", ds)
        hub_kw["cache_dir"] = hub
        ds_kw["cache_dir"] = ds
        print(f"HF cache: {hf_home}")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "nicholasKluge/TeenyTinyLlama-160m", **hub_kw
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "nicholasKluge/TeenyTinyLlama-160m", **hub_kw
    )
    
    # Load SuperReLoRA / ReLoRA model
    model = SuperReLoRaModel(
        base_model=base_model,
        r=8,
        alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        orthogonal_reinit=(args.method == "superrelora"),
    )
    
    # Load checkpoint (dir with safetensors/bin, or raw .pt/.bin)
    state_dict = load_state_dict(args.model_path, map_location="cpu")
    # U grows across merges (in_f, n*r); fresh model has (in_f, 0).
    # Forward does not use U — drop it so load_state_dict does not size-mismatch.
    dropped_u = [k for k in state_dict if k.endswith(".U")]
    if dropped_u:
        state_dict = {k: v for k, v in state_dict.items() if not k.endswith(".U")}
        print(f"Skipping {len(dropped_u)} orthonormal basis buffers (.U) for eval load")
    print("Checkpoint keys:", list(state_dict.keys())[:10])
    print("Model state_dict keys:", list(model.state_dict().keys())[:10])
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device)
    
    # Load and prepare dataset
    print("Loading dataset...")
    dataset = load_dataset(
        args.dataset_name, args.dataset_config, split='validation', **ds_kw
    )
    dataset = dataset.select(range(min(args.num_samples, len(dataset))))
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=args.max_length,
            padding='max_length'
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        tokenized_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Compute metrics
    print("Computing metrics...")
    loss, perplexity, accuracy = compute_perplexity(model, dataloader, device)
    print(f"\n{args.method} Model Metrics:")
    print(f"Loss: {loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    print(f"Accuracy: {accuracy:.4f}")

    metrics_out = args.metrics_out
    if metrics_out is None:
        # If model_path is .../runs/<method>/final_model, write sibling metrics.json
        parent = os.path.dirname(os.path.abspath(args.model_path.rstrip(os.sep)))
        if os.path.basename(args.model_path.rstrip(os.sep)) == "final_model":
            metrics_out = os.path.join(parent, "metrics.json")
    if metrics_out:
        payload = {
            "method": args.method,
            "loss": float(loss),
            "perplexity": float(perplexity),
            "accuracy": float(accuracy),
            "val_ppl": float(perplexity),
            "val_acc": float(accuracy),
            "num_samples": int(args.num_samples),
            "max_length": int(args.max_length),
            "model_path": args.model_path,
        }
        os.makedirs(os.path.dirname(os.path.abspath(metrics_out)) or ".", exist_ok=True)
        with open(metrics_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote metrics: {metrics_out}")
    
    # Generate example texts
    print("\nGenerating example texts:")
    test_prompts = [
        "Once upon a time",
        "The most important thing about",
        "In the future, artificial intelligence will",
    ]
    
    for prompt in test_prompts:
        generated = generate_text(model, tokenizer, prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated}")

if __name__ == '__main__':
    main() 