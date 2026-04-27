import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, ClassLabel
import os

# 1. Load Data
def load_data(file_path):
    print(f"Loading data from {file_path}...")
    try:
        # Assuming no header, columns: Sentiment, Sentence
        df = pd.read_csv(file_path, header=None, names=['sentiment', 'text'], encoding='ISO-8859-1')
        print(f"Loaded {len(df)} rows.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# 2. Preprocess
def preprocess_data(df, tokenizer):
    print("Preprocessing data...")
    
    # Map text labels to integers if needed
    # FinBERT labels: positive, negative, neutral (order matters based on model config)
    # ProsusAI/finbert config: 0: positive, 1: negative, 2: neutral (usually)
    # Let's check the model's config during load, but for now we map standard 3.
    
    # Standard FinBERT mapping usually: 0: positive, 1: negative, 2: neutral
    # But let's verify. We'll map to strings first then let the aligner handle it or assume standard.
    # Actually simpler: The model expects specific labels. 
    # Let's inspect the model config labels after loading.
    pass

def train():
    # Model ID
    model_id = "ProsusAI/finbert"
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load Model
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    
    # Check id2label
    id2label = model.config.id2label
    label2id = model.config.label2id
    print(f"Model Labels: {id2label}")
    
    # align dataset labels with model labels
    # Dataset labels in csv: 'positive', 'negative', 'neutral'
    # Model labels: usually 'positive', 'negative', 'neutral' keys
    
    df = load_data('all-data.csv')
    if df is None: return

    # Filter out any weird rows
    df = df.dropna()
    
    # Map labels
    # Ensure dataset sentiment matches label2id keys
    # Common issue: dataset might have 'neutral' but model expects 'neutral'
    
    def map_label(row_sentiment):
        return label2id.get(row_sentiment.lower(), -1)

    df['label'] = df['sentiment'].apply(map_label)
    
    # Drop rows where label couldn't be mapped
    df = df[df['label'] != -1]
    
    print("Converting to HuggingFace Dataset...")
    dataset = Dataset.from_pandas(df)
    
    # Split
    dataset = dataset.train_test_split(test_size=0.2)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Training Arguments
    # Optimized for CPU/Speed for demo (low epochs)
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,              # 1 Epoch for speed in this demo context
        per_device_train_batch_size=8,   # Low batch size for CPU
        per_device_eval_batch_size=8,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",              # Don't save intermediate checkpoints to save space
        use_cpu=not torch.cuda.is_available()
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test']
    )
    
    print("Starting Training (Fine-tuning)...")
    trainer.train()
    
    # Save Model
    save_path = "./saved_models/finbert_finai"
    print(f"Saving model to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Done!")

if __name__ == "__main__":
    train()
