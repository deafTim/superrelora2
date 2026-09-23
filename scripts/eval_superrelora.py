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
    parser = argparse.ArgumentParser(description="Evaluate SuperReLoRA / ReLoRA / LoRA model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint or final_model dir",
    )
    parser.add_argument("--dataset_name", type=str, default="Salesforce/wikitext", help="Dataset name")
    parser.add_argument(
        "--dataset_config", type=str, default="wikitext-2-raw-v1", help="Dataset config"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to evaluate")
    parser.add_argument(
        "--method",
        type=str,
        default="superrelora",
        choices=["superrelora", "relora", "lora"],
        help="Must match training method (orthogonal_reinit on/off)",
    )
    parser.add_argument(
        "--metrics_out",
        type=str,
        default=None,
        help="Optional path to write metrics.json",
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


def _batch_tensors(batch, device):
    def as_tensor(x):
        if torch.is_tensor(x):
            return x
        return torch.tensor(x)

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    if isinstance(input_ids, list):
        input_ids = torch.stack([as_tensor(x) for x in input_ids])
        attention_mask = torch.stack([as_tensor(x) for x in attention_mask])
    else:
        input_ids = as_tensor(input_ids)
        attention_mask = as_tensor(attention_mask)
    # Always [batch, seq] — never transpose based on shape comparison.
    if input_ids.ndim != 2:
        raise ValueError(f"expected 2D input_ids, got {tuple(input_ids.shape)}")
    return input_ids.to(device), attention_mask.to(device)


def compute_perplexity(model, dataloader, device, pad_token_id: int):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing metrics"):
            input_ids, attention_mask = _batch_tensors(batch, device)

            # Ignore pad positions in the LM loss (HF convention: labels == -100).
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            if pad_token_id is not None:
                labels[input_ids == pad_token_id] = -100

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            # Token-level mask for next-token positions (aligned with CE shift).
            token_mask = (labels[:, 1:] != -100).float()
            token_count = int(token_mask.sum().item())
            if token_count == 0:
                continue

            total_loss += float(loss.item()) * token_count
            total_tokens += token_count

            logits = outputs.logits
            predictions = torch.argmax(logits[:, :-1, :], dim=-1)
            target = labels[:, 1:]
            correct = ((predictions == target) & (target != -100)).sum().item()
            total_correct += int(correct)

    if total_tokens == 0:
        raise RuntimeError("No non-padding tokens in evaluation set")

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    accuracy = total_correct / total_tokens
    print(f"\nEvaluation complete! tokens={total_tokens} correct={total_correct}")
    return avg_loss, perplexity, accuracy


def generate_text(model, tokenizer, prompt, max_new_tokens=50):
    model.eval()
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    print("Loading model and tokenizer...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "nicholasKluge/TeenyTinyLlama-160m", **hub_kw
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "nicholasKluge/TeenyTinyLlama-160m", **hub_kw
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = SuperReLoRaModel(
        base_model=base_model,
        r=8,
        alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        orthogonal_reinit=(args.method == "superrelora"),
    )

    state_dict = load_state_dict(args.model_path, map_location="cpu")
    # U grows across merges; fresh model has U=(in_f, 0). Forward does not use U.
    dropped_u = [k for k in state_dict if k.endswith(".U")]
    if dropped_u:
        state_dict = {k: v for k, v in state_dict.items() if not k.endswith(".U")}
        print(f"Skipping {len(dropped_u)} orthonormal basis buffers (.U) for eval load")
    print("Checkpoint keys:", list(state_dict.keys())[:10])
    print("Model state_dict keys:", list(model.state_dict().keys())[:10])
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device)

    print("Loading dataset...")
    dataset = load_dataset(
        args.dataset_name, args.dataset_config, split="validation", **ds_kw
    )
    dataset = dataset.filter(lambda x: bool((x.get("text") or "").strip()))
    dataset = dataset.select(range(min(args.num_samples, len(dataset))))

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    dataloader = torch.utils.data.DataLoader(
        tokenized_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print("Computing metrics...")
    loss, perplexity, accuracy = compute_perplexity(
        model, dataloader, device, pad_token_id=tokenizer.pad_token_id
    )
    print(f"\n{args.method} Model Metrics:")
    print(f"Loss: {loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    print(f"Accuracy: {accuracy:.4f}")

    metrics_out = args.metrics_out
    if metrics_out is None:
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
        out_dir = os.path.dirname(os.path.abspath(metrics_out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(metrics_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote metrics: {metrics_out}")

    print("\nGenerating example texts:")
    for prompt in (
        "Once upon a time",
        "The most important thing about",
        "In the future, artificial intelligence will",
    ):
        generated = generate_text(model, tokenizer, prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated}")


if __name__ == "__main__":
    main()
