#!/usr/bin/env python3

"""
Train a DistilBERT model for news classification.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

from preprocess.preprocess import LABEL_MAP, MAX_LEN, label_df, load_data_labels

MODEL_NAME = "distilbert-base-uncased"

def split_data(texts, labels, size: float, seed:int): # data splitter from sklearn
    return train_test_split(texts, labels, test_size=size, random_state=seed, stratify=labels) #stratify for class balance

def build_dataset(texts: list[str], labels: list[int], tokenizer: DistilBertTokenizer):
    encodings = tokenizer(texts, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
    y = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(encodings["input_ids"], encodings["attention_mask"], y)

#handles batching and shuffling
def loader_make(text_train, text_val, labels_train, labels_val, tokenizer: DistilBertTokenizer, batch_size: int):
    train_loader = DataLoader(build_dataset(text_train, labels_train, tokenizer), batch_size=batch_size, shuffle=True) #randomize for train
    val_loader = DataLoader(build_dataset(text_val, labels_val, tokenizer), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

#1 epoch: for each batch, compute loss, backpropagate and update weights
def train_epoch(model, loader, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0
    for input_ids, attention_mask, labels in loader: # encoding input, enocoding attention, y (labels)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad() #stop gradient
        #supervision pass
        out = model(input_ids, attention_mask=attention_mask, labels=labels)
        out.loss.backward() #backpropagate
        optimizer.step() 

        total_loss += float(out.loss.item())
        total_batches += 1
    return total_loss / max(total_batches, 1)

def evaluate(model, loader, device: torch.device, *, report: bool = False) -> dict[str, float]:
    model.eval()
    all_predictions: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad(): # run without gradient computation
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device) 
            labels = labels.to(device)
            out = model(input_ids = input_ids, attention_mask=attention_mask, labels=labels).logits
            pred = out.argmax(dim=-1)
            all_predictions.extend(pred.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    if report:
        print(
            classification_report(
                all_labels,
                all_predictions,
                digits=4,
                zero_division=0,
                target_names=["FoxNews(0)", "NBC(1)"],
            )
        )

    return {
        "val accuracy": float(accuracy_score(all_labels, all_predictions)),
        "val f1 macro": float(f1_score(all_labels, all_predictions, average="macro", zero_division=0)),
        "val f1 weighted": float(
            f1_score(all_labels, all_predictions, average="weighted", zero_division=0)
        ),
    }

def checkpoint_save(model: DistilBertForSequenceClassification, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, path)
    print(f"Checkpoint saved to {path}")

def checkpoint_load(model: DistilBertForSequenceClassification, path: Path) -> DistilBertForSequenceClassification:
    model.load_state_dict(torch.load(path))
    model.eval()
    return model



def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="Newsheadlines/url_with_headlines.csv")
    p.add_argument("--out", type=str, default="checkpoints/model.pt")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--report", action="store_true")

    args = p.parse_args()

    texts, labels = load_data_labels(args.csv)
    text_train, text_val, labels_train, labels_val = split_data(texts, labels, args.val_size, args.seed)
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    train_loader, val_loader = loader_make(text_train, text_val, labels_train, labels_val, tokenizer, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use gpu if available
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(LABEL_MAP))
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1, verbose=True)

    best_val_f1 = float("-inf")

    for epoch in range(args.epochs):
        t_loss = train_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, val_loader, device, report=args.report)
        acc = metrics["val accuracy"]
        f1_macro = metrics["val f1 macro"]
        f1_weighted = metrics["val f1 weighted"]
        scheduler.step(f1_macro)
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {t_loss:.4f}, Val Accuracy: {acc:.4f}, Val F1 Macro: {f1_macro:.4f}, Val F1 Weighted: {f1_weighted:.4f}")
        if f1_macro > best_val_f1:
            best_val_f1 = f1_macro
            # checkpoint_save(model, args.out)
            print(f"New best F1 score: {best_val_f1:.4f}! Saving checkpoint to {args.out}")
        print("-"*50)
    print("Training complete!")
    checkpoint_save(model, Path(args.out))
    print(f"checkpoint saved to {args.out}")


if __name__ == "__main__":
    main()
    
