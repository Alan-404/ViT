import torch
from torchvision import transforms
import cv2 as cv
from PIL import Image
from typing import List, Union
import numpy as np

class ViTProcessor:
    def __init__(self,
                 input_size: Union[List[int], int],
                 labels: List[str]) -> None:
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else input_size

        self.resize = transforms.Resize(self.input_size)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.labels = labels
    
    def load_image(self, path: str) -> torch.Tensor:
        image = Image.open(path).convert('RGB')
        image = self.resize(image)
        image = torch.tensor(np.array(image)).transpose(0, 2)

        return image

    def as_target(self, labels: List[str]):
        ids = []
        for label in labels:
            ids.append(self.labels.index(label))
        return torch.tensor(ids)
    
    def __call__(self, images: List[torch.Tensor]) -> torch.Any:
        images = torch.stack(images)
        return self.normalize(images)
    