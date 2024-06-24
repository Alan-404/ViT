import torch
import torch.nn.functional as F

import numpy as np

from typing import Union, List

class ViTCriterion:
    def __init__(self) -> None:
        pass
    def cross_entropy_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        F.cross_entropy(logits, labels)

class ViTMetric:
    def __init__(self) -> None:
        pass

    def accuracy_score(self, preds: Union[torch.Tensor, np.ndarray, List[str], List[int]], labels: Union[torch.Tensor, np.ndarray, List[str], List[int]]):
        if isinstance(preds, List):
            preds = np.array(preds)
            labels = np.array(labels)
        elif isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()

        return np.count_nonzero(preds == labels) / len(preds)