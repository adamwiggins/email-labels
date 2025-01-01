from typing import Protocol
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import sqlite3
import re
import pandas as pd

class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class DistilBertProvider:
    def __init__(self, model_path: str = "distilbert-base-uncased"):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=3  # inbox, fyi, junk
        )
        self.label_map = {'inbox': 0, 'fyi': 1, 'junk': 2}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}

    def preprocess_text(self, text):
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        # Normalize spaces
        text = re.sub(r'\s+', ' ', text).strip()
        # Lowercase
        text = text.lower()
        return text

    def get_completion(self, content: str, prompt: str) -> str:
        content = self.preprocess_text(content)
        inputs = self.tokenizer(content, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**inputs)
        predicted_label_id = torch.argmax(outputs.logits, dim=1).item()
        return self.reverse_label_map[predicted_label_id]
    
    def fine_tune(self, db_path: str = "datasets/for-finetuning.sqlite", 
                  epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5):
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT sender_name, sender_email, subject, body, label FROM labeled_emails", conn)
        conn.close()
        
        # Combine subject and body for training
        texts = [f"From: {row['sender_name']} <{row['sender_email']}>\nSubject: {row['subject']}\n\n{row['body']}" for _, row in df.iterrows()]
        # Strip out HTML etc
        texts = [self.preprocess_text(text) for text in texts]
        
        labels = [self.label_map[label.lower()] for label in df['label']]
        
        dataset = EmailDataset(texts, labels, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup training
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.model.train()
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")
        
        # Add these lines to save the model and tokenizer
        output_dir = "fine_tuned_model"
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model and tokenizer saved to {output_dir}/")

if __name__ == "__main__":
    provider = DistilBertProvider()
    provider.fine_tune(
        db_path="datasets/for-finetuning.sqlite",
        epochs=3,
        batch_size=16,
        learning_rate=2e-5
    )