import math
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate base model')
    parser.add_argument('--dataset_name', type=str, default='wikitext', help='Dataset name')
    parser.add_argument('--dataset_config', type=str, default='wikitext-2-raw-v1', help='Dataset config')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--max_length', type=int, default=64, help='Max sequence length')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to evaluate')
    return parser.parse_args()

def compute_metrics(model, dataloader, device):
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
            
            # Ensure [batch, seq]
            if input_ids.shape[0] < input_ids.shape[1]:
                input_ids = input_ids.T
                attention_mask = attention_mask.T
            
            print(f"\nProcessing batch {batch_idx + 1}/{num_batches}")
            print(f"Batch shape: {input_ids.shape}")
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(0)
            
            # Compute accuracy
            logits = outputs.logits  # [batch, seq, vocab]
            predictions = torch.argmax(logits, dim=-1)  # [batch, seq]
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
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
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
    print(f"\nUsing device: {device}")
    
    # Load base model and tokenizer
    print("\nLoading base model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained("nicholasKluge/TeenyTinyLlama-160m")
    tokenizer = AutoTokenizer.from_pretrained("nicholasKluge/TeenyTinyLlama-160m")
    model = model.to(device)
    print("Model loaded successfully!")
    
    # Load and prepare dataset
    print("\nLoading dataset...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
    dataset = dataset.select(range(min(args.num_samples, len(dataset))))
    print(f"Dataset loaded with {len(dataset)} examples")
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=64,  # Shorter sequences
            padding='max_length'
        )
    
    print("\nTokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    print("Tokenization complete!")
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        tokenized_dataset,
        batch_size=16,  # Larger batch size for faster processing
        shuffle=False
    )
    print(f"Created dataloader with {len(dataloader)} batches")
    
    # Compute metrics
    print("\nStarting metrics computation...")
    loss, perplexity, accuracy = compute_metrics(model, dataloader, device)
    print(f"\nBase Model Metrics (Quick Evaluation):")
    print(f"Loss: {loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Generate example texts
    print("\nGenerating example texts:")
    test_prompts = [
        "Once upon a time",
        "The most important thing about",
    ]
    
    for prompt in test_prompts:
        generated = generate_text(model, tokenizer, prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated}")
        print("-" * 50)

if __name__ == '__main__':
    main() 