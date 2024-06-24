import torch
import torch.nn as nn

from ..utils.block import EncoderBlock

from typing import Optional

class Encoder(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, activation: str = 'gelu', dropout_p: float = 0.) -> None:
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(d_model, n_heads, activation, dropout_p) for _ in range(n_layers)])
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None)-> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x