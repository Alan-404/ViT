import os

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from torch.cuda.amp import GradScaler, autocast

from .model.vit import ViT
from .processing.processor import ViTProcessor
from .dataset import ViTCollate, ViTDataset
from .evaluation import ViTCriterion

from tqdm import tqdm
from typing import Optional

def setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', world_size=world_size, rank=rank)
    print(f"Initialize Thread {rank+1}/{world_size}")

def cleanup():
    dist.destroy_process_group()

def train(rank: int,
          world_size: int,
          # Data Train Config
          train_path: str,
          num_epochs: int = 1,
          train_batch_size: int = 1,
          num_train_samples: Optional[int] = None,
          lr: float = 7e-5,
          fp16: bool = True,
          # Data Val Config
          val_path: Optional[str] = None,
          val_batch_size: int = 1,
          num_val_samples: Optional[int] = None,
          # Processor Config
          input_size: int = 64,
          label_path: str = "./configs/cifar10.json",
          #
          patch_size: int = 16,
          n_layers: int = 12, 
          d_model: int = 768, 
          n_heads: int = 12, 
          activation: str = 'gelu', 
          dropout_p: float = 0.1
        ):
    processor = ViTProcessor(input_size, label_path)

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

    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    collate_fn = ViTCollate(processor=processor)
    
    train_dataset = ViTDataset(train_path, processor=processor, num_examples=num_train_samples)
    train_sampler = DistributedSampler(dataset=train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else RandomSampler(train_dataset)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, sampler=train_sampler, collate_fn=collate_fn)

    scaler = GradScaler(enabled=fp16)
    criterion = ViTCriterion()

    for epoch in range(num_epochs):
        train_losses = []

        for index, (x, y) in enumerate(tqdm(train_dataloader)):
            with autocast(enabled=fp16):
                outputs = model(x)
            loss = criterion.cross_entropy_loss(outputs, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)

            scaler.update()

            train_losses.append(loss.item())
        
        

