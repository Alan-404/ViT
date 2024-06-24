import torch
from torch.utils.data import Dataset

from processing.processor import ViTProcessor

import pandas as pd
from typing import Optional, Tuple, List

class ViTDataset(Dataset):
    def __init__(self, manifest_path: str, processor: ViTProcessor, num_examples: Optional[int] = None) -> None:
        super().__init__()
        self.prompts = pd.read_csv(manifest_path)
        if num_examples is not None:
            self.prompts = self.prompts[:num_examples]

        self.processor = processor
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        index_df = self.prompts.iloc[index]

        image = self.processor.load_image(index_df['path'])
        label = index_df['label']

        return image, label
    
class ViTCollate:
    def __init__(self, processor: ViTProcessor) -> None:
        self.processor = processor

    def __call__(self, batch: Tuple[List[torch.Tensor], List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        images, labels = zip(*batch)

        images = self.processor(images)
        labels = self.processor.as_target(labels)

        return images, labels