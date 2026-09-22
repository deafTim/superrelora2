import os
import sys

os.environ["WANDB_MODE"] = "disabled"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse

import torch
import torch.optim as optim
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from src.superrelora_model import SuperReLoRaModel
from src.utils import save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train SuperReLoRA / ReLoRA / LoRA / full FT")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--use_trainer", action="store_true", help="Use HuggingFace Trainer")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument(
        "--method",
        type=str,
        choices=["superrelora", "relora", "lora", "full"],
        help="Training method (overrides config)",
    )
    parser.add_argument("--merge_every", type=int, help="Steps between merge-and-reinit")
    parser.add_argument("--max_steps", type=int, help="Maximum number of training steps")
    parser.add_argument("--logging_steps", type=int, help="Logging steps")
    parser.add_argument("--eval_steps", type=int, help="Evaluation steps")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs")
    parser.add_argument("--limit_train_examples", type=int, help="Limit train examples (debug)")
    parser.add_argument(
        "--use_base_model",
        action="store_true",
        help="Deprecated alias for --method full",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_method(config, args) -> str:
    if args.use_base_model:
        return "full"
    if args.method is not None:
        return args.method
    return config.get("method", "superrelora")


def setup_hf_cache(config) -> str | None:
    """Put HuggingFace hub/datasets cache on GPFS (or any path in config/env)."""
    cache = (
        config.get("hf_home")
        or os.environ.get("SUPERRELORA_HF_HOME")
        or os.environ.get("HF_HOME")
    )
    if not cache:
        return None
    cache = os.path.expanduser(cache)
    os.makedirs(cache, exist_ok=True)
    hub = os.path.join(cache, "hub")
    datasets_cache = os.path.join(cache, "datasets")
    transformers_cache = os.path.join(cache, "transformers")
    os.makedirs(hub, exist_ok=True)
    os.makedirs(datasets_cache, exist_ok=True)
    os.makedirs(transformers_cache, exist_ok=True)
    os.environ["HF_HOME"] = cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = hub
    os.environ["TRANSFORMERS_CACHE"] = transformers_cache
    os.environ["HF_DATASETS_CACHE"] = datasets_cache
    print(f"HF cache (model + datasets): {cache}")
    return cache


def _hub_kwargs():
    hub = os.environ.get("HUGGINGFACE_HUB_CACHE")
    return {"cache_dir": hub} if hub else {}


def prepare_model_and_tokenizer(config, method: str):
    hub_kw = _hub_kwargs()
    # Prefer fp16 on Turing GPUs (e.g. RTX 2080 Ti); bf16 often breaks / is slow.
    use_bf16 = bool(config.get("bf16", False))
    use_fp16 = bool(config.get("fp16", True)) and not use_bf16
    torch_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)
    hub_kw = {**hub_kw, "torch_dtype": torch_dtype}

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], **{k: v for k, v in hub_kw.items() if k != "torch_dtype"})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if method == "full":
        model = AutoModelForCausalLM.from_pretrained(config["model_name"], **hub_kw)
        return model, tokenizer

    base = AutoModelForCausalLM.from_pretrained(config["model_name"], **hub_kw)
    orthogonal = method == "superrelora"
    # lora: wrap but never merge; relora/superrelora: merge
    model = SuperReLoRaModel(
        base_model=base,
        r=config["lora_r"],
        alpha=config["lora_alpha"],
        target_modules=config.get("target_modules", []),
        orthogonal_reinit=orthogonal,
        prune_ratio=float(config.get("prune_ratio", 0.99)),
    )
    return model, tokenizer


def prepare_dataset(config, tokenizer):
    ds_kwargs = {}
    if os.environ.get("HF_DATASETS_CACHE"):
        ds_kwargs["cache_dir"] = os.environ["HF_DATASETS_CACHE"]
    dataset = load_dataset(
        config["dataset_name"],
        config.get("dataset_config", None),
        split="train",
        **ds_kwargs,
    )
    limit = config.get("limit_train_examples", None)
    if limit is not None and limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config["max_length"],
            padding="max_length",
        )

    return dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )


class MergeReinitCallback(TrainerCallback):
    """Periodic merge-and-reinit for ReLoRA / SuperReLoRa under HF Trainer."""

    def __init__(self, merge_every: int, base_lr: float, warmup_steps: int = 50):
        self.merge_every = merge_every
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self._warmup_left = 0

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        if self.merge_every <= 0 or model is None:
            return
        if not hasattr(model, "step_merge_reinit"):
            return

        step = state.global_step
        norm = model.step_merge_reinit(step=step, every=self.merge_every, optimizer=optimizer)
        if norm > 0 and optimizer is not None:
            # Jagged LR: drop to 0 and warm back up
            self._warmup_left = self.warmup_steps
            for group in optimizer.param_groups:
                group["lr"] = 0.0
            print(f"[merge] step={step} delta_norm={norm:.4f} (orthogonal={model.orthogonal_reinit})")

        if self._warmup_left > 0 and optimizer is not None:
            done = self.warmup_steps - self._warmup_left + 1
            scale = min(1.0, done / max(1, self.warmup_steps))
            for group in optimizer.param_groups:
                group["lr"] = self.base_lr * scale
            self._warmup_left -= 1


def train_with_trainer(model, tokenizer, dataset, config, output_dir, method: str):
    config["learning_rate"] = float(config["learning_rate"])
    config["weight_decay"] = float(config["weight_decay"])
    config["batch_size"] = int(config["batch_size"])

    merge_every = int(config.get("merge_every", 0) or 0)
    if method == "lora":
        merge_every = 0

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["num_epochs"],
        max_steps=config.get("max_steps", -1) if config.get("max_steps") else -1,
        per_device_train_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        logging_steps=int(config.get("logging_steps", 100)),
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to=[],
        fp16=bool(config.get("fp16", True)) and not bool(config.get("bf16", False)),
        bf16=bool(config.get("bf16", False)),
    )

    callbacks = []
    if merge_every > 0 and isinstance(model, SuperReLoRaModel):
        callbacks.append(
            MergeReinitCallback(
                merge_every=merge_every,
                base_lr=config["learning_rate"],
                warmup_steps=int(config.get("jagged_warmup_steps", 50)),
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=callbacks,
    )
    trainer.train()
    trainer.save_model(os.path.join(output_dir, "final_model"))


def _batch_to_device(batch, device):
    def to_tensor(x):
        if torch.is_tensor(x):
            return x
        return torch.tensor(x)

    input_ids = to_tensor(batch["input_ids"])
    attention_mask = to_tensor(batch["attention_mask"])
    if isinstance(batch["input_ids"], list):
        input_ids = torch.stack([to_tensor(x) for x in batch["input_ids"]])
        attention_mask = torch.stack([to_tensor(x) for x in batch["attention_mask"]])
    return input_ids.to(device), attention_mask.to(device)


def train_manual(model, tokenizer, dataset, config, output_dir, method: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    lr = float(config["learning_rate"])
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=float(config["weight_decay"]),
    )

    merge_every = int(config.get("merge_every", 0) or 0)
    if method == "lora":
        merge_every = 0
    jagged_warmup = int(config.get("jagged_warmup_steps", 50))
    warmup_left = 0

    dataloader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=True)
    log_path = os.path.join(output_dir, "loss_log.csv")
    with open(log_path, "w") as f:
        f.write("step,epoch,loss,merged\n")

    max_steps = config.get("max_steps", None)
    global_step = 0

    for epoch in range(int(config["num_epochs"])):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}")

        for step, batch in enumerate(progress_bar):
            input_ids, attention_mask = _batch_to_device(batch, device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            merged = 0.0
            global_step += 1
            if isinstance(model, SuperReLoRaModel) and merge_every > 0:
                merged = model.step_merge_reinit(
                    step=global_step, every=merge_every, optimizer=optimizer
                )
                if merged > 0:
                    warmup_left = jagged_warmup
                    for group in optimizer.param_groups:
                        group["lr"] = 0.0
                    print(
                        f"[merge] step={global_step} delta_norm={merged:.4f} "
                        f"(orthogonal={model.orthogonal_reinit})"
                    )

            if warmup_left > 0:
                done = jagged_warmup - warmup_left + 1
                scale = min(1.0, done / max(1, jagged_warmup))
                for group in optimizer.param_groups:
                    group["lr"] = lr * scale
                warmup_left -= 1

            progress_bar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})
            with open(log_path, "a") as f:
                f.write(f"{global_step},{epoch + 1},{loss.item():.6f},{merged:.6f}\n")

            if max_steps is not None and global_step >= int(max_steps):
                break

        avg_loss = total_loss / max(1, len(dataloader))
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            path=os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}.pt"),
            extra_state={"method": method, "global_step": global_step},
        )
        if max_steps is not None and global_step >= int(max_steps):
            break

    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pt"))


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.merge_every is not None:
        config["merge_every"] = args.merge_every
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if args.logging_steps is not None:
        config["logging_steps"] = args.logging_steps
    if args.eval_steps is not None:
        config["eval_steps"] = args.eval_steps
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.num_epochs is not None:
        config["num_epochs"] = args.num_epochs
    if args.limit_train_examples is not None:
        config["limit_train_examples"] = args.limit_train_examples

    method = resolve_method(config, args)
    config["method"] = method
    os.makedirs(args.output_dir, exist_ok=True)

    setup_hf_cache(config)

    model, tokenizer = prepare_model_and_tokenizer(config, method=method)
    dataset = prepare_dataset(config, tokenizer)

    print("method:", method)
    if isinstance(model, SuperReLoRaModel):
        print("orthogonal_reinit:", model.orthogonal_reinit)
        print("replaced modules:", len(model.replaced_modules))
    print("CUDA available:", torch.cuda.is_available())
    print("device:", next(model.parameters()).device)
    print("dtype:", next(model.parameters()).dtype)

    if args.use_trainer:
        train_with_trainer(model, tokenizer, dataset, config, args.output_dir, method)
    else:
        train_manual(model, tokenizer, dataset, config, args.output_dir, method)


if __name__ == "__main__":
    main()
