from typing import Optional, Callable

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_masked_attention_mask(seq_len: str):
    mask = torch.ones(size=(1, seq_len, seq_len), dtype=torch.uint8)
    mask = mask.triu(diagonal=1)
    return mask == 0

class MultiHeadedAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        h: int = 8,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        dropout: float=0.1
    ):
        super(MultiHeadedAttention, self).__init__()
        self.h = h
        self.d_model = d_model

        if not d_k:
            assert self.d_model % h == 0
            self.d_k = self.d_model // h
        else:
            self.d_k = d_k

        self.d_q = self.d_k
        self.d_v = d_v or self.d_k

        self.Q = nn.Linear(self.d_model, self.d_q * h)
        self.K = nn.Linear(self.d_model, self.d_k * h)
        self.V = nn.Linear(self.d_model, self.d_v * h)
        self.O = nn.Linear(self.d_v * h, self.d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.attn = None # For visualization of attention mechanism

    def self_attention(
        self, query: torch.tensor, key: torch.tensor, value: torch.tensor, mask=None, dropout=None
    ):
        # Query & Key: (batch_size, n_head, seq_len, d_k) -> Key Transpose = (batch_size, n_head, d_k, seq_len)
        # Value: (batch_size, n_head, seq_len, d_v)
        # Mask: (batch_size, 1, seq_len, seq_len)
        d_k = query.size(-1)
        self_attention = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) # (batch_size, n_head, seq_len, seq_len)
        if mask is not None:
            self_attention = self_attention.masked_fill(mask==0, value=1e-9)
        attention_weights = self_attention.softmax(dim=-1) # (batch_size, n_head, seq_len, seq_len)
        if dropout:
            attention_weights = dropout(attention_weights)
        return torch.matmul(attention_weights, value), attention_weights # (batch_size, n_head, seq_len, d_v)

    def forward(
        self, query, key, value, mask=None
    ):
        # Encoder & Decoder attention unit: Query = Key = Value = (batch_size, seq_len, emb_size)
        # Decoder masked attention unit: Query = (batch_size, tgt_len, emb_size) , K = V = (batch_size, src_len, emb_size)
        # Mask: For masked attention = (batch_size, padded_tgt_len, padded_tgt_len)
        # Mask: For self attention = (1, 1 tgt_len/src_len)

        if mask is not None:
            mask = mask.unsqueeze(1) # (batch_size, 1, tgt_len, tgt_len)

        batch_size = query.size(0)

        query = self.Q(query) # (batch_size, seq_len, d_q * h)
        query = query.view(batch_size, -1, self.h, self.d_q) # (batch_size, seq_len, h, d_q)
        query = query.transpose(1, 2) # (batch_size, h, seq_len, d_q)

        key = self.K(key) # (batch_size, seq_len, d_k * h)
        key = key.view(batch_size, -1, self.h, self.d_k)
        key = key.transpose(1, 2)

        value = self.V(value) # (batch_size, seq-len, d_v * h)
        value = value.view(batch_size, -1, self.h, self.d_v)
        value = value.transpose(1, 2) # (batch_size, h, value_seq_len, )

        x, self.attn = self.self_attention(query, key, value, mask, self.dropout) # x = (batch_size, n_head, seq_len, d_v)
        x = x.transpose(1,2).contiguous().view(batch_size, -1, self.h * self.d_v)
        output = self.O(x) # (batch_size, seq_len, d_model)

        del query
        del key
        del value

        return output

class PointWiseFeedForward(nn.Module):
    def __init__(
        self, d_model: int=512, d_ff: Optional[int] = None, dropout: float=0.1
    ):
        super(PointWiseFeedForward, self).__init__()
        self.d_model=d_model
        self.d_ff = d_ff or 4 * d_model
        self.linear1 = nn.Linear(self.d_model, self.d_ff)
        self.linear2 = nn.Linear(self.d_ff, self.d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Input batch: (batch_size, seq-len, d_model)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class LayerNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float=1e-8
        ):
        super(LayerNorm, self).__init__()
        self.scaling_factor = nn.Parameter(torch.ones(d_model))
        self.bias_factor = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # (batch_size, seq_len, d_model)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        activations = (x - mean) / (std + self.eps)
        return self.scaling_factor * activations + self.bias_factor # Point-wise additional and multiplication

class SubLayerConnection(nn.Module):
    def __init__(
        self, d_model: int, eps: float=1e-6, dropout: float=0.1
    ):
        super(SubLayerConnection, self).__init__()
        self.d_model = d_model
        self.layer_norm = LayerNorm(d_model=d_model, eps=eps)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self, x: torch.Tensor, sub_layer: Callable
    ):
        return self.layer_norm(x + self.dropout(sub_layer(x)))