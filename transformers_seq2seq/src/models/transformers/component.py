import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List
from copy import deepcopy as dc
from .block import SubLayerConnection, PointWiseFeedForward, MultiHeadedAttention

def clones(module: nn.Module, N: List):
    return nn.ModuleList([dc(module) for _ in range(N)])

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, d_model: int=512, h: int=8
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.multihead_attention = MultiHeadedAttention(d_model=d_model, h=h)
        self.feed_forward = PointWiseFeedForward(d_model=d_model)
        self.att_sublayer = SubLayerConnection(d_model=d_model)
        self.ff_sublayer = SubLayerConnection(d_model=d_model)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor]=None
    ):
        x = self.att_sublayer(x, lambda x: self.multihead_attention(query=x, key=x, value=x, mask=mask))
        x = self.ff_sublayer(x, self.feed_forward)
        return x

class TransformerEncoder(nn.Module):
    def __init__(
        self, n: int=6, h: int=8, d_model: int=512
    ):
        super(TransformerEncoder, self).__init__()
        self.N, self.H, self.d_model = n, h, d_model
        self.layers = clones(TransformerEncoderLayer(d_model=d_model, h=self.H), self.N)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor]=None
    ):
        # Input = (batch_size, seq_len, d_model)
        # Ouput = (batch_size, seq_len, d_model)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self, d_model: int=512, h: int=8
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.d_model = d_model
        self.masked_multihead_attention = MultiHeadedAttention(d_model=d_model, h=h)
        self.multihead_attention = MultiHeadedAttention(d_model=d_model, h=h)
        self.feed_forward = PointWiseFeedForward(d_model=d_model)
        self.masked_att_sublayer = SubLayerConnection(d_model=d_model)
        self.att_sublayer = SubLayerConnection(d_model=d_model)
        self.ff_sublayer = SubLayerConnection(d_model=d_model)

    def forward(
        self,
        decoder_input: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor]=None,
        tgt_mask: Optional[torch.Tensor]=None
    ):
        # Decoder input: (bs, tgt_seq_len, d_model)
        # Encoder ouput: (bs, src_seq_len, d_model)
        x = self.masked_att_sublayer(
            decoder_input,
            lambda x: self.masked_multihead_attention(query=x, key=x, value=x, mask=tgt_mask)
            )
        x = self.att_sublayer(
            x,
            lambda x: self.multihead_attention(query=x, key=encoder_output, value=encoder_output, mask=src_mask)
            )
        x = self.ff_sublayer(x, self.feed_forward)
        return x

class TransformerDecoder(nn.Module):
    def __init__(
        self, n: int=6, h: int=8, d_model: int=512
    ):
        super(TransformerDecoder, self).__init__()
        self.N, self.H, self.d_model = n, h, d_model
        self.layers = clones(TransformerDecoderLayer(d_model=d_model, h=self.H), self.N)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor]=None,
        tgt_mask: Optional[torch.Tensor]=None
    ):
        # x: (bs, tgt_seq_len, d_model)
        # encoder_output: (bs, src_seq_len, d_model)
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x
    
class Generator(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, temperature: float = 1.0):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        self.temperature = temperature
        
    def forward(self, x: torch.tensor, temperature: Optional[float] = None):
        temperature = temperature or self.temperature
        return F.log_softmax(self.proj(x / self.temperature), dim=-1)