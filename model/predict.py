#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

from preprocess.preprocess import LABEL_MAP, MAX_LEN, MODEL_NAME

LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}


def load_model(checkpoint_path: str, device: torch.device) -> DistilBertForSequenceClassification:
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(LABEL_MAP))
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.to(device)
    model.eval()
    return model


def predict(headlines: list[str], checkpoint_path: str, device: torch.device | None = None) -> list[dict]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = load_model(checkpoint_path, device)

    encodings = tokenizer(
        headlines,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

    results = []
    for i, headline in enumerate(headlines):
        label_idx = int(preds[i].item())
        results.append({
            "headline": headline,
            "label": LABEL_NAMES[label_idx],
            "confidence": round(float(probs[i][label_idx].item()), 4),
        })
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="Path to saved .pt checkpoint")
    p.add_argument("--headlines", type=str, nargs="+", help="One or more headline strings")
    p.add_argument("--csv", type=str, help="CSV file with a 'headline' column to run batch predictions on")
    p.add_argument("--output", type=str, help="Save batch predictions to this CSV path")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv)
        headlines = df["headline"].astype(str).tolist()
        results = predict(headlines, args.checkpoint, device)
        out_df = pd.DataFrame(results)
        if args.output:
            out_df.to_csv(args.output, index=False)
            print(f"Saved {len(out_df)} predictions to {args.output}")
        else:
            print(out_df.to_string(index=False))
    elif args.headlines:
        results = predict(args.headlines, args.checkpoint, device)
        for r in results:
            print(f"[{r['label']} {r['confidence']:.2%}] {r['headline']}")
    else:
        p.error("Provide --headlines or --csv")


if __name__ == "__main__":
    main()
