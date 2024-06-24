import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

class CheckpointManager:
    def __init__(self, saved_folder: str, num_saved: int = 3) -> None:
        self.saved_folder = saved_folder
        self.num_saved = num_saved

        self.saved_checkpoints = []

    def load_checkpoint(self, checkpoint_path: str, model: nn.Module, optimizer: optim.Optimizer, scheduler: lr_scheduler.LRScheduler):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        n_steps = checkpoint['n_steps']
        n_epochs = checkpoint['n_epochs']

        return n_steps, n_epochs
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, scheduler: lr_scheduler.LRScheduler, n_steps: int, n_epochs: int):
        data = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'n_steps': n_steps,
            'n_epochs': n_epochs
        }

        if len(self.saved_checkpoints) == self.num_saved:
            shutil.rmtree(f"{self.saved_folder}/{self.saved_checkpoints[0]}.pt")
            self.saved_checkpoints.pop(0)

        torch.save(data, f"{self.saved_folder}/{n_steps}.pt")

        self.saved_checkpoints(n_steps)