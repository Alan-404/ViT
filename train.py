import os

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .model.vit import ViT
from .processing.processor import ViTProcessor

from typing import Optional

def setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', world_size=world_size, rank=rank)
    print(f"Initialize Thread {rank+1}/{world_size}")

def train(rank: int,
          world_size: int,
          # Data Train Config
          train_path: str,
          num_epochs: int = 1,
          train_batch_size: int = 1,
          num_train_samples: Optional[int] = None,
          # Data Val Config
          val_path: Optional[str] = None,
          val_batch_size: int = 1,
          num_val_samples: Optional[int] = None,
          # Processor Config
          ):
    pass