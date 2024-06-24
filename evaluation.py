import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score

class ViTCriterion:
    def __init__(self) -> None:
        pass
    def cross_entropy_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        F.cross_entropy(logits, labels)

class ViTMetric:
    def __init__(self) -> None:
        pass

    def accuracy(self, preds: torch.Tensor, labels: torch.Tensor):
        accuracy_score(labels.numpy(), preds.numpy())