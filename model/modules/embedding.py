import torch
import torch.nn as nn

from typing import Tuple

class PatchEmbedding(nn.Module):
    def __init__(self, n_samples_per_patch: Tuple[int, int], patch_size: Tuple[int, int], d_model: int, input_channels: int = 3) -> None:
        super().__init__()
        self.n_patches_per_height, self.n_patches_per_width = n_samples_per_patch
        self.patch_height, self.patch_width = patch_size

        self.patch_dim = input_channels * self.patch_width * self.patch_height

        self.input_channels = input_channels

        self.norm_1 = nn.LayerNorm(self.patch_dim)
        self.linear = nn.Linear(self.patch_dim, d_model)
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        '''
            x: input, shape = [batch_size, input_channels, width, height]
            Require: 
            - width = n_patches_per_width * patch_width
            - height = n_patches_per_height * patch_height
        '''
        '''
            Convert shape of input from [..., input_channels, input_width, input_height] --> [..., n_samples, patch_dim]
            With: 
            - n_samples = n_patches_per_width * n_patches_per_height 
            - patch_dim = input_channels * patch_width * patch_height
            - input_width = n_patches_per_width * patch_width
            - input_height = n_patches_per_height * patch_height
        '''

        batch_size = x.size(0)
        
        x = x.reshape((batch_size, self.input_channels, self.n_patches_per_height, self.patch_height, self.n_patches_per_width, self.patch_width))
        x = x.permute([0, 2, 4, 1, 3, 5]) # shape = [batch_size, n_patches_per_width, n_patches_per_height, channels, patch_width, patch_height]
        x = x.reshape((batch_size, self.n_patches_per_width * self.n_patches_per_height, self.patch_dim))

        x = self.norm_1(x)
        x = self.linear(x)
        x = self.norm_2(x)
        
        return x