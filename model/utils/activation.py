import torch.nn as nn

activations = nn.ModuleDict({
    'relu': nn.ReLU(),
    'gelu': nn.GELU(),
    'selu': nn.SELU(),
    'silu': nn.SiLU()
})