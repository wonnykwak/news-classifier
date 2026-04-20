import torch
from torch import nn
from typing import Any, Iterable, List
from transformers import DistilBertForSequenceClassification


class Model(nn.Module):
    """
    Template model for the leaderboard.

    Requirements:
    - Must be instantiable with no arguments (called by the evaluator).
    - Must implement `predict(batch)` which receives an iterable of inputs and
      returns a list of predictions (labels).
    - Must implement `eval()` to place the model in evaluation mode.
    - If you use PyTorch, submit a state_dict to be loaded via `load_state_dict`
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=2,
        )
        self.model.to(self.device)
        self.model.eval()

    def eval(self) -> "Model":
        super().eval()
        self.model.eval()
        return self

    def predict(self, batch: Iterable[Any]) -> List[Any]:
        """
        Implement your inference here.
        Inputs:
            batch: Iterable of preprocessed inputs (as produced by your preprocess.py)
        Returns:
            A list of predictions with the same length as `batch`.
        """
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
    """
    Factory function required by the evaluator.
    Returns an uninitialized model instance. The evaluator may optionally load
    weights (if provided) before calling predict(...).
    """
    return Model()


