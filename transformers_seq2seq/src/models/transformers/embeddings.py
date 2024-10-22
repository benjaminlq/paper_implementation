import torch
import torch.nn as nn
import math

class TokenEmbeddings(nn.Module):
    def __init__(
        self, d_model: int, vocab_size: int
    ):
        super(TokenEmbeddings, self).__init__()
        self.d_model, self.vocab_size = d_model, vocab_size
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model, padding_idx=0
        ) 
    
    def forward(
        self, inputs: torch.tensor
    ):
        return self.embeddings(inputs) * math.sqrt(self.d_model)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(
        self, d_model: int, dropout: float=0.1, max_len: int=5000
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
        self, x: torch.tensor
    ):
        assert x.size(2) == self.d_model
        # Input sequence: (batch_size, seq_length)
        x = x + self.pe[:, : x.size(1), :].requires_grad_(False)
        return self.dropout(x)

class LearnablePositionalEncoding(nn.Module):
    def __init__(
        self, d_model: int, dropout: float=0.1, max_len: int=5000
    ):
        super(LearnablePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.embeddings = nn.Embedding(max_len, d_model)

    def forward(
        self, x
    ):
        assert x.size(2) == self.d_model
        embs = self.embeddings(torch.arange(0, x.size(1), 1))
        x = x + embs
        return self.dropout(x)