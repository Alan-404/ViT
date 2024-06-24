import os

import torch
from torch.utils.data import DataLoader

from .processing.processor import ViTProcessor
from .model.vit import ViT
from .dataset import ViTDataset, ViTCollate
from .evaluation import ViTMetric
from .manager import CheckpointManager

from tqdm import tqdm
from typing import Optional

import fire

def test(
        rank: int,
        world_size: int,
        test_path: str,
        checkpoint: str,
        batch_size: int = 1,
        num_samples: Optional[int] = None,
        # Processor Config
        input_size: int = 64,
        label_path: str = "./configs/cifar10.json",
        # Model Config
        patch_size: int = 16,
        n_layers: int = 12, 
        d_model: int = 768, 
        n_heads: int = 12, 
        activation: str = 'gelu', 
        dropout_p: float = 0.0,
        # Result config
        saved_path: str = "./results/result.csv"
    ):

    assert os.path.exists(test_path) and os.path.exists(checkpoint)

    checkpoint_manager = CheckpointManager()

    processor = ViTProcessor(
        input_size=input_size,
        labels=label_path
    )

    model = ViT(
        n_classes=len(processor.labels),
        input_channels=3,
        input_size=input_size,
        patch_size=patch_size,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        activation=activation,
        dropout_p=dropout_p
    ).to(rank)

    checkpoint_manager.load_model(checkpoint, model)

    dataset = ViTDataset(test_path, processor, num_examples=num_samples)
    collate_fn = ViTCollate(processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    metric = ViTMetric()
    preds = []
    scores = []

    model.eval()
    for (x, y) in tqdm(dataloader):
        with torch.inference_mode():
            outputs = model(x)
        
        predictions = torch.max(outputs, dim=-1)
        scores.append(metric.accuracy(predictions, y))
        for prediction in predictions:
            preds.append(processor.idx_to_label(prediction.cpu().numpy()))

    dataset.prompts['prediction'] = preds
    dataset.prompts.to_csv(saved_path, index=False)


def main():
    pass

if __name__ == '__main__':
    fire.Fire(main)