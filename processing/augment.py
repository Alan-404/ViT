import torch
from torchvision import transforms

class ViTAugment:
    def __init__(self) -> None:
        self.pipeline = transforms.RandomHorizontalFlip()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pipeline(x)
        return x