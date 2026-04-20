import re
import torch
from typing import Any, List, Tuple
import pandas as pd
from transformers import DistilBertTokenizer

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

LABEL_MAP = {"FoxNews": 0, "NBC": 1}

def prepare_data(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Template preprocessing for leaderboard.

    Requirements:
    - Must read the provided data path at `path`.
    - Must return a tuple (X, y):
        X: a list of model-ready inputs (these must match what your model expects in predict(...))
        y: a list of ground-truth labels aligned with X (same length)

    Notes:
    - The evaluation backend will call this function with the shared validation data
    - Ensure the output format (types, shapes) of X matches your model's predict(...) inputs.
    """
    df = pd.read_csv(path)
    
    # add source column to dataframe
    df["source"] = df["url"].apply(lambda url: "FoxNews" if "foxnews" in str(url) else "NBC")


    # remove null headline
    df = df.dropna(subset=["headline"]).reset_index(drop=True)

    headlines = df["headline"]

    # label: FoxNews=0, NBC=1
    y = df["source"].apply(lambda x: LABEL_MAP.get(str(x).strip(), 0)).tolist()

    # 토크나이징, 샘플별 dict 리스트로 반환
    encodings = tokenizer(
        headlines,
        max_length=MAX_LEN,
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

    # raise NotImplementedError("Implement prepare_data(csv_path) -> (X, y).")