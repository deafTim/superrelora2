import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoConfig
)
from datasets import load_dataset
import yaml
from tqdm import tqdm

from src.superrelora_model import SuperReLoRaModel
from src.utils import save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description='Train SuperReLoRA model')
    parser.add_argument('--config', type=str, required=True, help='Path to training config YAML')
    parser.add_argument('--use_trainer', action='store_true', help='Use HuggingFace Trainer')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--merge_every', type=int, help='Steps between partial merges')
    parser.add_argument('--merge_alpha', type=float, help='Merge coefficient alpha')
    parser.add_argument('--max_steps', type=int, help='Maximum number of training steps')
    parser.add_argument('--logging_steps', type=int, help='Logging steps')
    parser.add_argument('--eval_steps', type=int, help='Evaluation steps')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def prepare_model_and_tokenizer(config):
    model = AutoModelForCausalLM.from_pretrained(config['model_name'])
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    # Wrap model with SuperReLoRA
    model = SuperReLoRaModel(
        base_model=model,
        r=config['lora_r'],
        alpha=config['lora_alpha'],
        target_modules=config.get('target_modules', [])
    )
    
    return model, tokenizer

def prepare_dataset(config, tokenizer):
    # Load dataset
    dataset = load_dataset(config['dataset_name'], split='train')
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=config['max_length'],
            padding='max_length'
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def train_with_trainer(model, tokenizer, dataset, config, output_dir):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config['num_epochs'],
        per_device_train_batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=100,
        save_strategy='epoch',
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )
    
    trainer.train()
    trainer.save_model(os.path.join(output_dir, 'final_model'))

def train_manual(model, tokenizer, dataset, config, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    log_path = os.path.join(output_dir, 'loss_log.csv')
    with open(log_path, 'w') as f:
        f.write('step,epoch,loss\n')
    
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{config["num_epochs"]}')
        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            progress_bar.set_postfix({'loss': loss.item()})
            
            # log to CSV
            global_step = epoch * len(dataloader) + step
            with open(log_path, 'a') as f:
                f.write(f"{global_step},{epoch+1},{loss.item():.6f}\n")
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1} average loss: {avg_loss:.4f}')
        
        # Save checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            path=os.path.join(output_dir, f'checkpoint_epoch_{epoch + 1}.pt')
        )
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pt'))

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare model and dataset
    model, tokenizer = prepare_model_and_tokenizer(config)
    dataset = prepare_dataset(config, tokenizer)
    
    # Train model
    if args.use_trainer:
        train_with_trainer(model, tokenizer, dataset, config, args.output_dir)
    else:
        train_manual(model, tokenizer, dataset, config, args.output_dir)

    if args.merge_every is not None:
        config['merge_every'] = args.merge_every
    if args.merge_alpha is not None:
        config['merge_alpha'] = args.merge_alpha
    if args.max_steps is not None:
        config['max_steps'] = args.max_steps
    if args.logging_steps is not None:
        config['logging_steps'] = args.logging_steps
    if args.eval_steps is not None:
        config['eval_steps'] = args.eval_steps

if __name__ == '__main__':
    main() 