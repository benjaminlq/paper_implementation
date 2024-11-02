import torch
import torch.nn as nn
import math
from typing import Optional

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(
        self, d_model: int, dropout: float=0.1, max_len: int=512
    ):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(size=(max_len, d_model)) # (max_len, d_model)
        const_term = math.log(10000) / d_model
        div_terms = torch.exp(-torch.arange(0, d_model, 2) * const_term) # (d_model//2)
        positions = torch.arange(0, max_len).unsqueeze(1) # (max_len, 1)
        pe[:, ::2] = torch.sin(positions*div_terms)  # sin(pos * div_term)
        pe[:, 1::2] = torch.cos(positions*div_terms)  # sin(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(
        self, inputs: torch.Tensor
    ):
        # Input sequence: (batch_size, seq_length)
        seq_len = inputs.size(1)
        x = self.pe[:, : seq_len, :].requires_grad_(False)
        return self.dropout(x)

class TokenEmbeddings(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int=30522,
        padding_idx: int=0
    ):
        super(TokenEmbeddings, self).__init__()
        self.d_model, self.vocab_size = d_model, vocab_size
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model, padding_idx=padding_idx
        )

    def forward(
        self, inputs: torch.Tensor
    ):
        return self.embeddings(inputs) * math.sqrt(self.d_model)

class SegmentEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        padding_idx: int = 2,
    ):
        super(SegmentEmbedding, self).__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(
                3, embedding_dim=d_model, padding_idx=padding_idx
            )

    def forward(
        self,
        token_types_id: torch.Tensor, # (batch_size, seq_len)
        attn_mask: Optional[torch.Tensor] = None
        ):
        token_types_id = token_types_id.masked_fill(
            attn_mask==0, self.padding_idx
        )
        segment_embeddings = self.embedding(token_types_id)
        return segment_embeddings # (batch_size, seq_len, d_model)

class BERTEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        vocab_size: int = 30522,
        use_segment_embedding: bool = False,
        dropout: float = 0.1,
        token_padding_idx: int = 0,
        segment_padding_idx: int = 2
    ):
        super(BERTEmbedding, self).__init__()
        self.token_embedding = TokenEmbeddings(
            d_model=d_model,
            vocab_size=vocab_size,
            padding_idx=token_padding_idx
            )
        self.positional_embedding = SinusoidalPositionalEncoding(
            d_model=d_model, max_len=512
        )
        self.segment_embedding = None
        if use_segment_embedding:
            self.segment_embedding = SegmentEmbedding(
                d_model=d_model, padding_idx=segment_padding_idx
            )
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        inputs: torch.Tensor, # (batch_size, seq_len)
        token_type_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        token_embeddings = self.token_embedding(inputs)
        pos_embeddings = self.positional_embedding(inputs)
        embeddings = token_embeddings + pos_embeddings
        if self.segment_embedding:
            embeddings = embeddings + self.segment_embedding(token_type_ids, attn_mask)
        return self.dropout(embeddings)