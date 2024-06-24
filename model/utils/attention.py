import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Optional

class MultiHeadAttention(nn.Module):
    '''
        Implementation from "Attention Is All You Need"
        arXiv: https://arxiv.org/pdf/1706.03762
        @article{vaswani2017attention,
            title={Attention is all you need},
            author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
            journal={Advances in neural information processing systems},
            volume={30},
            year={2017}
        }
    '''
    def __init__(self, d_model: int, n_heads: int, dropout_p: float = 0.) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout_p = dropout_p

        self.n_head_samples = d_model // n_heads
        self.sqrt_dim = math.sqrt(self.n_head_samples)

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)

        self.linear_output = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''
            q: query --> shape = [batch_size, n_heads, query_time, n_head_samples],
            k, v: key, value --> shape = [batch_size, n_heads, cross_time, n_head_samples]
            mask:
            - Padding Mask: shape = [batch_size, 1, 1, cross_time]
            - Look Ahead Mask: shape = [batch_size]
            --------------------------------------------------------
            Attention = softmax((Q x K^T) / sqrt(d_k)) x V
            Process: Matmul --> Scale --> (Optional) Mask -> Softmax --> Matmul
        '''
        # Compute Attention Score = (q x k^T) / sqrt(d_k) and Scale
        attention_score = torch.matmul(q, k.transpose(-1, -2)) # shape = [batch_size, n_heads, query_time, cross_time]
        attention_score = attention_score / self.sqrt_dim
        
        # (Optional) Masking
        if mask is not None:
            attention_score.masked_fill_(mask, 1e-4) # 1e-4 is better for fp16 setup than negative infinity

        # Softmax Activation
        attention_weights = F.softmax(attention_score, dim=-1)
        attention_weights = F.dropout(attention_weights, p=self.dropout_p, training=self.training)

        # Matmul
        attention_context = torch.matmul(attention_weights, v) # shape = [batch_size, n_heads, query_time, n_head_samples]
        
        return attention_context
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''
            q: query --> shape = [batch_size, query_time, d_model],
            k, v: key, value --> shape = [batch_size, cross_time, d_model]
            mask:
            - Padding Mask: shape = [batch_size, 1, 1, cross_time]
            - Look Ahead Mask: shape = [batch_size]
            --------------------------------------------------------
            Attention = softmax((Q x K^T) / sqrt(d_k)) x V
        '''
        batch_size, query_time, _ = q.size()
        cross_time = k.size(1)

        # Projection
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        # Split Heads
        q = q.reshape((batch_size, query_time, self.n_heads, self.n_head_samples)).transpose(1, 2) # shape = [batch_size, n_heads, query_time, n_head_samples]
        k = k.reshape((batch_size, cross_time, self.n_heads, self.n_head_samples)).transpose(1, 2) # shape = [batch_size, n_heads, cross_time, n_head_samples]
        v = v.reshape((batch_size, cross_time, self.n_heads, self.n_head_samples)).transpose(1, 2) # shape = [batch_size, n_heads, cross_time, n_head_samples]

        # Scaled-dot Product Attention
        attention_context = self.scaled_dot_product_attention(q, k, v, mask) # shape = [batch_size, n_heads, query_time, n_head_samples]

        # Final Projection
        attention_context = attention_context.transpose(1, 2).reshape((batch_size, query_time, self.d_model))
        attention_context = self.linear_output(attention_context)

        return attention_context