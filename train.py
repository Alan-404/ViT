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
from .evaluation import ViTCriterion, ViTMetric
from .manager import CheckpointManager

import statistics
from tqdm import tqdm
import fire
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
          # Checkpoint Config
          checkpoint: Optional[str] = None,
          saved_checkpoints: str = "./checkpoints",
          save_checkpoint_after: int = 1,
          # Data Val Config
          val_path: Optional[str] = None,
          val_batch_size: int = 1,
          num_val_samples: Optional[int] = None,
          # Processor Config
          input_size: int = 64,
          label_path: str = "./configs/cifar10.json",
          # Model Config
          patch_size: int = 16,
          n_layers: int = 12, 
          d_model: int = 768, 
          n_heads: int = 12, 
          activation: str = 'gelu', 
          dropout_p: float = 0.1
        ):
    if rank == 0:
        if os.path.exists(saved_checkpoints) == False:
            os.makedirs(saved_checkpoints)

        checkpoint_manager = CheckpointManager(saved_checkpoints)
        n_steps = 0
        n_epochs = 0

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

    if checkpoint is not None:
        assert os.path.exists(checkpoint), "NOT FOUND CHECKPOINT"
        n_steps, n_epochs = checkpoint_manager.load_checkpoint(checkpoint, model, optimizer, scheduler)

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    collate_fn = ViTCollate(processor=processor)
    
    train_dataset = ViTDataset(train_path, processor=processor, num_examples=num_train_samples)
    train_sampler = DistributedSampler(dataset=train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else RandomSampler(train_dataset)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, sampler=train_sampler, collate_fn=collate_fn)

    is_validation = val_path is not None and os.path.exists(val_path)

    if is_validation:
        val_dataset = ViTDataset(val_path, processor=processor, num_examples=num_val_samples)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else RandomSampler(val_dataset)
        val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, sampler=val_sampler, collate_fn=collate_fn)

    scaler = GradScaler(enabled=fp16)
    criterion = ViTCriterion()
    metric = ViTMetric()

    for epoch in range(num_epochs):
        train_losses = []

        if is_validation:
            val_losses = []
            val_scores = []
        
        model.train()
        for (x, y) in tqdm(train_dataloader, leave=False):
            with autocast(enabled=fp16):
                outputs = model(x)
            loss = criterion.cross_entropy_loss(outputs, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)

            scaler.update()

            train_losses.append(loss.item())

            n_steps += 1
        
        if is_validation:
            model.eval()
            for (x, y) in tqdm(val_dataloader, leave=False):
                with torch.no_grad():
                    with autocast(enabled=fp16):
                        outputs = model(x)
                    loss = criterion.cross_entropy_loss(outputs, y).item()
                    preds = torch.argmax(outputs, dim=-1)
                    score = metric.accuracy(preds.cpu().numpy(), y.cpu().numpy())

                    val_losses.append(loss)
                    val_scores.append(score)
                    
        n_epochs += 1

        if rank == 0:
            train_loss = statistics.mean(train_losses)
            print(f"Train Loss: {(train_loss):.4f}")
            if is_validation:
                val_loss = statistics.mean(val_losses)
                val_score = statistics.mean(val_scores)

                print(f"Val Loss: {(val_loss):.4f}")
                print(f"Val Score: {(val_score):.4f}")

            if epoch % save_checkpoint_after == save_checkpoint_after - 1 or epoch == num_epochs - 1:
                checkpoint_manager.save_checkpoint(model, optimizer, scheduler, n_steps, n_epochs)

    if world_size > 1:
        cleanup()

def main(
        # Data Train Config
          train_path: str,
          num_epochs: int = 1,
          train_batch_size: int = 1,
          num_train_samples: Optional[int] = None,
          lr: float = 2e-4,
          fp16: bool = True,
          # Checkpoint Config
          checkpoint: Optional[str] = None,
          saved_checkpoints: str = "./checkpoints",
          save_checkpoint_after: int = 1,
          # Data Val Config
          val_path: Optional[str] = None,
          val_batch_size: int = 1,
          num_val_samples: Optional[int] = None,
          # Processor Config
          input_size: int = 64,
          label_path: str = "./configs/cifar10.json",
          # Model Config
          patch_size: int = 16,
          n_layers: int = 12, 
          d_model: int = 768, 
          n_heads: int = 12, 
          activation: str = 'gelu', 
          dropout_p: float = 0.1
        ):
    n_gpus = torch.cuda.device_count()

    if n_gpus == 1:
        train(
            0, n_gpus, train_path, num_epochs, train_batch_size, num_train_samples, lr, bool(fp16),
            checkpoint, saved_checkpoints, save_checkpoint_after,
            val_path, val_batch_size, num_val_samples,
            input_size, label_path,
            patch_size, n_layers, d_model, n_heads, activation, dropout_p
        )
    elif n_gpus > 1:
        mp.spawn(
            train,
            args=(
                n_gpus, train_path, num_epochs, train_batch_size, num_train_samples, lr, bool(fp16),
                checkpoint, saved_checkpoints, save_checkpoint_after,
                val_path, val_batch_size, num_val_samples,
                input_size, label_path,
                patch_size, n_layers, d_model, n_heads, activation, dropout_p
            ),
            nprocs=n_gpus,
            join=True
        )
    else:
        print("NOT SUPPORT FOR CPU")

if __name__ == '__main__':
    fire.Fire(main)