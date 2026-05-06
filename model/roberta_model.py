import torch
from torch import nn
from typing import Any, Iterable, List
import os
from transformers import AutoModelForSequenceClassification


class Model(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        model_name = "roberta-base"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
        )
        self.model.to(self.device)
        self.model.eval()

    def eval(self) -> "Model":
        super().eval()
        self.model.eval()
        return self

    def predict(self, batch: Iterable[Any]) -> List[Any]:
        preds: List[Any] = []
        self.model.eval()

        with torch.no_grad():
            for item in batch:
                if not isinstance(item, dict):
                    raise TypeError(
                        "Each batch item must be a dict like "
                        "{'input_ids': Tensor, 'attention_mask': Tensor}."
                    )

                input_ids = item["input_ids"].to(self.device)
                attention_mask = item["attention_mask"].to(self.device)

                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                if attention_mask.dim() == 1:
                    attention_mask = attention_mask.unsqueeze(0)

                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits
                pred = int(torch.argmax(logits, dim=-1).item())
                preds.append(pred)

        return preds


def get_model() -> Model:
    return Model()


