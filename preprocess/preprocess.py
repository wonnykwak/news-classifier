import torch
from typing import Tuple
import pandas as pd
import os
from transformers import DistilBertTokenizer

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = int(os.getenv("MAX_LEN", "32"))

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

LABEL_MAP = {"FoxNews": 0, "NBC": 1}

def label_df(path:str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["source"] = df["url"].apply(lambda url: "FoxNews" if "foxnews" in str(url).lower() else "NBC")
    df = df.dropna(subset=["headline"]).reset_index(drop=True)
    df["headline"] = df["headline"].astype(str)
    return df

def load_data_labels(path:str) -> tuple[list[str], list[int]]:
    df = label_df(path)
    text = df["headline"].astype(str).tolist()
    labels = df["source"].apply(lambda x: LABEL_MAP.get(str(x).strip(), 0)).tolist()
    return text, labels

def prepare_data(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    df = label_df(path)

    headlines = df["headline"].astype(str).tolist()
    # label: FoxNews=0, NBC=1
    y = df["source"].apply(lambda x: LABEL_MAP.get(str(x).strip(), 0)).tolist()

    encodings = tokenizer(
        headlines,
        max_length=MAX_LEN, # 64
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    X = [{
        "input_ids":      encodings["input_ids"][i],
        "attention_mask": encodings["attention_mask"][i],
    }
        for i in range(len(headlines))
         ]

    return X, y

