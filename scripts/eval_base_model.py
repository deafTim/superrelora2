import math
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm

def compute_metrics(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing metrics"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
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
            predictions = predictions[..., :-1, :].contiguous()
            labels = input_ids[..., 1:].contiguous()
            mask = attention_mask[..., 1:].contiguous()
            
            # Count correct predictions
            correct = (predictions == labels) * mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    accuracy = total_correct / total_tokens
    
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load base model and tokenizer
    print("Loading base model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained("nicholasKluge/TeenyTinyLlama-160m")
    tokenizer = AutoTokenizer.from_pretrained("nicholasKluge/TeenyTinyLlama-160m")
    model = model.to(device)
    
    # Load and prepare dataset
    print("Loading dataset...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
    dataset = dataset.select(range(min(1000, len(dataset))))  # Use same number of samples as SuperReLoRA eval
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=128,
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
        batch_size=8,
        shuffle=False
    )
    
    # Compute metrics
    print("Computing metrics...")
    loss, perplexity, accuracy = compute_metrics(model, dataloader, device)
    print(f"\nBase Model Metrics:")
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