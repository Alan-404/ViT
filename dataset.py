import torch
from torch.utils.data import Dataset

from .processing.processor import ViTProcessor

import pandas as pd
from typing import Optional, Tuple, List

class ViTDataset(Dataset):
    def __init__(self, manifest_path: str, processor: ViTProcessor, training: bool = False, num_examples: Optional[int] = None) -> None:
        super().__init__()
        self.prompts = pd.read_csv(manifest_path)
        if num_examples is not None:
            self.prompts = self.prompts[:num_examples]

        self.processor = processor
        self.training = training
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        index_df = self.prompts.iloc[index]

        image = self.processor.load_image(index_df['path'])

        if self.training:
            label = index_df['label']
            return image, label
        else:
            return image
    
class ViTCollate:
    def __init__(self, processor: ViTProcessor, training: bool = False) -> None:
        self.processor = processor
        self.training = training

    def __call__(self, batch: Tuple[List[torch.Tensor], List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            images, labels = zip(*batch)
            images = self.processor(images)
            labels = self.processor.as_target(labels)
            return images, labels
        else:
            images = self.processor(batch)
            return images