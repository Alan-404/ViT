import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.embedding import PatchEmbedding
from model.modules.encoder import Encoder

from typing import Union, List, Tuple

class ViT(nn.Module):
    '''
        Implemention from "AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE"
        arXiv: https://arxiv.org/pdf/2010.11929
        @article{dosovitskiy2020image,
            title={An image is worth 16x16 words: Transformers for image recognition at scale},
            author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
            journal={arXiv preprint arXiv:2010.11929},
            year={2020}
        }
    '''
    def __init__(self, 
                 n_classes: int, 
                 input_channels: int = 3,
                 input_size: Union[List[int], Tuple[int], int] = (64, 64), 
                 patch_size: Union[List[int], Tuple[int], int] = (16, 16), 
                 n_layers: int = 12, 
                 d_model: int = 768, 
                 n_heads: int = 12, 
                 activation: str = 'gelu', 
                 dropout_p: float = 0.,
                 pool: str = 'cls') -> None:
        super().__init__()
        # Input and Patch Size Config
        image_height, image_width = (input_size, input_size) if isinstance(input_size, int) else input_size
        patch_height, patch_width = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

        num_patches_per_width = image_width // patch_width
        num_patches_per_height = image_height // patch_height

        num_patches = num_patches_per_width * num_patches_per_height

        # Head Config
        self.dropout_p = dropout_p
        self.pool = pool

        # Patch Embedding
        self.patch_embedding = PatchEmbedding((num_patches_per_height, num_patches_per_width), patch_size, d_model, input_channels)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
        
        # Transformer Encoder
        self.encoder = Encoder(n_layers, d_model, n_heads, activation, dropout_p)

        # Multi - Layer Perceptron Head
        self.latent = nn.Identity()
        self.mlp_head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
            x: shape = [batch_size, channels = 3, width, height]
        '''
        # Patch Embedding
        x = self.patch_embedding(x)

        # Connect with CLS Token and Add Position Information
        x = torch.cat([self.cls_token, x], dim=1)
        x = x + self.pos_embedding
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        # Transformer Encoder
        x = self.encoder(x)

        # Head
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.latent(x)
        x = self.mlp_head(x)

        return x
