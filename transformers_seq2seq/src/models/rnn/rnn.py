import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal, Tuple

class RNNEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int, emb_size: int, hidden_size: int,
        num_layers: int = 2, dropout: float = 0.0,
        rnn_type: Literal["LSTM", "RNN", "GRU"] = "GRU",
        batch_first: bool = True, bidirectional: bool = True
    ):
        super(RNNEncoder, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=emb_size, padding_idx=0
        )
        self.rnn = getattr(nn, rnn_type)(
            input_size=emb_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=batch_first,
            dropout=dropout, bidirectional=bidirectional
            )
        self.dropout = nn.Dropout(p=dropout)
        
        self._bidirectional = bidirectional
        self._rnn_type = rnn_type
        self._hidden_size = hidden_size
        self._num_layers = num_layers 
        
    def forward(
        self,
        inputs: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor]:
        ## Input: (bs, enc_seq_length)
        embs = self.dropout(self.embeddings(inputs)) # (bs, enc_seq_length, enc_vocab_size)
        if self._rnn_type == "lstm":
            outputs, (hidden, _) = self.rnn(embs) # (bs, enc_seq_length, hidden_size), (no_directions * num_layers, bs, hidden_size), (no_directions * num_layers, bs, hidden_size)
        else:
            outputs, hidden = self.rnn(embs) # (bs, seq_length, hidden_size), (no_directions * num_layers, bs, hidden_size)
            
        return outputs, hidden
        
class RNNDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int, emb_size: int, hidden_size: int,
        num_layers: int = 2, dropout: float = 0.0,
        rnn_type: Literal["LSTM", "RNN", "GRU"] = "GRU",
        batch_first: bool = True, use_attention: bool = True
    ):
        super(RNNDecoder, self).__init__()
        self._use_attention = use_attention
        self._rnn_type = rnn_type
        self._hidden_size = hidden_size
        self._num_layers = num_layers 
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=emb_size, padding_idx=0
        )
        self.rnn = getattr(nn, rnn_type)(
            input_size=emb_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=batch_first, dropout=dropout,
            bidirectional=False
            )
        self.dropout = nn.Dropout(p = dropout)
        if self._use_attention:
            self.output = nn.Linear(hidden_size * 2, vocab_size)
        else:
            self.output = nn.Linear(hidden_size, vocab_size)

    def attention(
        self, 
        encoder_hiddens: torch.tensor, decoder_hidden: torch.tensor
        ) -> torch.tensor:
        ## Encoder Hiddens: (bs, enc_len, hidden_size)
        ## Decoder hidden: (bs, hidden_size)
        att_weight = torch.bmm(encoder_hiddens, decoder_hidden.unsqueeze(2)) # Att_weight = (bs, enc_len, 1)
        att_weight = F.softmax(att_weight.squeeze(2), dim = 1) # Att_weight = (bs, enc_len)

        att_output = torch.bmm(encoder_hiddens.transpose(1,2), att_weight.unsqueeze(2)).squeeze(2) # (bs, hidden_size)
        att_combined = torch.cat((att_output, decoder_hidden), dim = 1) # (bs, hidden_size * 2)

        return att_combined

    def forward(
        self,
        inputs: torch.tensor,
        encoder_hiddens: torch.tensor,
        encoder_final_hidden: torch.tensor,
    ) -> Tuple[torch.tensor, torch.tensor]:
        ## Inputs: (bs)
        ## Encoder_hiddens: (bs, enc_seq, hidden_size)
        inputs = inputs.unsqueeze(1) # (bs, 1)
        x = self.dropout(self.emb(inputs)) # (bs, 1, dec_vocab_size)
        
        x, hidden = self.rnn(x, encoder_final_hidden) # (bs, 1, enc_hidden_size), (bs, no_of_directions * num_layers, hidden_size)
        if self._use_attention:
            x = self.attention(encoder_hiddens, x.squeeze(1)) # (bs, hidden_size * 2)
        x = self.output(x) # (bs, dec_vocab_size)
        out = F.log_softmax(x, dim = 1) # (bs, dec_vocab_size)
        return out, hidden # (bs, dec_vocab_size), (bs, no_of_directions * num_layers, hidden_size)