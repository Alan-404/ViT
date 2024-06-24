import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention
from .activation import activations
from typing import Optional

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, activation: str = 'gelu', dropout_p: float = 0.) -> None:
        super().__init__()
        self.dropout_p = dropout_p

        # Main Layers
        self.attention = MultiHeadAttention(d_model, n_heads, dropout_p)
        self.mlp = MLP(d_model, activation)

        # Norm Layers
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Sub-layer 1
        attention = self.attention(x, x, x, mask)
        attention = self.norm_1(F.dropout(attention + x, p=self.dropout_p, training=self.training))

        # Sub-layer 2
        mlp_out = self.mlp(attention)
        mlp_out = self.norm_2(F.dropout(mlp_out + attention, p=self.dropout_p, training=self.training))

        return mlp_out

class MLP(nn.Module):
    '''
        Implementation by "Attention is All you Need" but change the activation from ReLU to GELU based on "AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE"
    '''
    def __init__(self, d_model: int, activation: str = 'gelu', n_expands: int = 4) -> None:
        super().__init__()
        activation = activation.lower()
        assert activation in activations.keys()
        hidden_dim = d_model * n_expands

        self.hidden_layer = nn.Linear(d_model, hidden_dim)
        self.activation = activations[activation]
        self.final_layer = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.final_layer(x)
        return x