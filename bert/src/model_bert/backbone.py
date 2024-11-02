import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
from model_bert.embedding import BERTEmbedding

class GroupAttentionHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_q: int,
        d_k: int,
        d_v: int,
        n_heads: int = 1,
        dropout: float = 0.1
    ):
        super(GroupAttentionHead, self).__init__()
        self.d_q, self.d_k, self.d_v = d_q, d_k, d_v
        self.Q = nn.ModuleList([nn.Linear(d_model, d_q) for _ in range(n_heads)])
        self.K = nn.Linear(d_model, d_k)
        self.V = nn.Linear(d_model, d_v)
        self.dropout = dropout

    def forward(
        self,
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        attn_mask: Optional[torch.Tensor]=None,
        dropout: Optional[float]=None,
        is_causal: bool=False
    ):
        """Calculate attention values for a single attention head.
        For self attention, q = k = v; Dimension = (batch_size, seq_len, d_model)
        For seq2seq cross attention
        - q = output of masked attention previous component of the decoder. Dimension = (batch_size, tgt_seq_len, d_model)
        - k = v = output of the encoder. Dimension = (batch_size, src_seq_len, d_model)
        """
        dropout = dropout or self.dropout
        query = torch.cat(
            [q(query).unsqueeze(1) for q in self.Q], dim = 1
        ) # (batch_size, n_groups, seq_len, d_q)
        key = self.K(key).unsqueeze(1) # (batch_size, 1, seq_len, d_k)
        value = self.V(value).unsqueeze(1) # (batch_size, 1, seq_len, d_v)
        if isinstance(attn_mask, torch.Tensor):
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1).type(torch.bool)

        values = F.scaled_dot_product_attention(
            query, key, value, attn_mask, dropout, is_causal
            )

        return values

class MultiHeadedAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        n_groups: int = 12,
        dropout: float = 0.1,
        d_v: Optional[int] = None
    ):
        super(MultiHeadedAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_groups = n_groups
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_groups == 0, "Total number of heads must be divisible by number of groups"
        self.heads_per_group = n_heads // n_groups
        self.d_q = d_model // n_heads
        self.d_k = self.d_q
        self.d_v = d_v or self.d_q
        self.attn_heads = nn.ModuleList(
            [
                GroupAttentionHead(d_model, self.d_q, self.d_k, self.d_v, n_heads=self.heads_per_group, dropout=dropout)
                for _ in range(n_groups)
                ]
            )
        self.O = nn.Linear(self.d_v * self.n_heads, self.d_model)

    def forward(
        self,
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        attn_mask: Optional[torch.Tensor]=None,
        dropout: Optional[float]=None,
        is_causal: bool=False
    ):
        batch_size = query.size(0)
        tgt_seq_len = value.size(1)
        values = torch.cat(
            [
                att_head(
                    query, key, value,
                    attn_mask=attn_mask, dropout=dropout, is_causal=is_causal)
                for att_head in self.attn_heads
                ], dim = 1
            ).transpose(1, 2) # (bs, tgt_seq_len, n_heads * n_groups, d_v)
        values = values.contiguous().view(batch_size, tgt_seq_len, -1)
        return self.O(values)

class PointwiseFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        d_feedfoward: Optional[int] = None,
        activation: str = "gelu",
    ):
        super(PointwiseFeedForward, self).__init__()
        self.d_feedforward = d_feedfoward or d_model * 4
        self.linear1 = nn.Linear(d_model, self.d_feedforward)
        self.linear2 = nn.Linear(self.d_feedforward, d_model)
        self.activation = getattr(nn.functional, activation)

    def forward(
        self, inputs: torch.Tensor
    ):
        x = self.linear1(inputs)
        x = self.linear2(self.activation(x))
        return x

class ResidualLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        dropout: float = 0.1
    ):
        super(ResidualLayer, self).__init__()
        self.d_model = d_model
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model, eps=1e-5)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self, inputs: torch.Tensor, sublayer: Callable
    ):
        return self.layer_norm(inputs + self.dropout(sublayer(inputs)))

class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int=768,
        n_heads: int=12,
        n_groups: int=3,
        dropout: float=0.1,
        d_v: Optional[int] = None,
        d_feedfoward: Optional[int] = None,
        activation: str = "gelu",
    ):
        super(EncoderBlock, self).__init__()
        self.d_model = d_model
        self.multihead_group_attention = MultiHeadedAttention(
            d_model=d_model, n_heads=n_heads, n_groups=n_groups, dropout=dropout, d_v=d_v,
        )
        self.feedforward = PointwiseFeedForward(d_model=d_model, d_feedfoward=d_feedfoward, activation=activation)
        self.att_residual = ResidualLayer(d_model=d_model)
        self.ff_residual = ResidualLayer(d_model=d_model)

    def forward(
        self,
        inputs: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ):
        x = self.att_residual(
            inputs,
            lambda x: self.multihead_group_attention(
                x, x, x, attn_mask=attn_mask, is_causal=is_causal
            )
        )
        x = self.ff_residual(x, self.feedforward)
        return x

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int=768,
        n_layers: int=12,
        n_heads: int=12,
        n_groups: int=3,
        dropout: float=0.1,
        d_v: Optional[int] = None,
        d_feedfoward: Optional[int] = None,
        activation: str = "gelu",
    ):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=d_model, n_heads=n_heads, n_groups=n_groups, dropout=dropout, d_v=d_v,
                    d_feedfoward=d_feedfoward, activation=activation)
                for _ in range(n_layers)
                ]
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ):
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, is_causal=is_causal)
        return x

class BERTBackbone(nn.Module):
    def __init__(
        self,
        d_model: int=768,
        vocab_size: int=30522,
        n_layers: int=12,
        n_heads: int=12,
        n_groups: int=3,
        dropout: float=0.1,
        d_v: Optional[int] = None,
        d_feedfoward: Optional[int] = None,
        activation: str = "gelu",
        use_segment_embedding: bool = False
    ):
        super(BERTBackbone, self).__init__()

        self.embedding = BERTEmbedding(
            d_model=d_model,
            vocab_size=vocab_size,
            use_segment_embedding=use_segment_embedding,
            dropout=dropout
        )

        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_groups=n_groups,
            dropout=dropout,
            d_v=d_v,
            d_feedfoward=d_feedfoward,
            activation=activation
        )

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Initialize weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        inputs: torch.Tensor, # (batch_size, seq_len)
        token_type_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ):
        x = self.embedding(
            inputs=inputs, token_type_ids=token_type_ids, attn_mask=attn_mask
        )
        x = self.encoder(
            x,
            attn_mask=attn_mask.type(torch.float),
            is_causal=is_causal
            )
        return x

