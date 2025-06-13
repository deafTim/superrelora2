import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import math
import argparse
from src.superrelora_model import SuperReLoRaModel

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate SuperReLoRA model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset_name', type=str, default='wikitext', help='Dataset name')
    parser.add_argument('--dataset_config', type=str, default='wikitext-2-raw-v1', help='Dataset config')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--max_length', type=int, default=128, help='Max sequence length')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to evaluate')
    return parser.parse_args()

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
            total_loss += loss.item() * input_ids.size(0)
            
            # Compute accuracy
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            # Shift predictions and labels for next token prediction
            predictions = predictions[:, :-1].contiguous()
            labels = input_ids[:, 1:].contiguous()
            mask = attention_mask[:, 1:].contiguous()
            
            # Count correct predictions
            correct = (predictions == labels) * mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
            
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
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    base_model = AutoModelForCausalLM.from_pretrained("nicholasKluge/TeenyTinyLlama-160m")
    tokenizer = AutoTokenizer.from_pretrained("nicholasKluge/TeenyTinyLlama-160m")
    
    # Load SuperReLoRA model
    model = SuperReLoRaModel(
        base_model=base_model,
        r=8,
        alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    print("Checkpoint keys:", list(checkpoint.keys())[:10])
    print("Model state_dict keys:", list(model.state_dict().keys())[:10])
    model.load_state_dict(checkpoint)
    model = model.to(device)
    
    # Load and prepare dataset
    print("Loading dataset...")
    dataset = load_dataset(args.dataset_name, args.dataset_config, split='validation')
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
    print(f"\nSuperReLoRA Model Metrics:")
    print(f"Loss: {loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    print(f"Accuracy: {accuracy:.4f}")
    
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