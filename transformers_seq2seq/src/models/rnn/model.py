import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal, Callable
import config
import random

class SeqToSeq(nn.Module):
    def __init__(
        self,
        encoder: Callable,
        decoder: Callable,
    ):
        """Sequence To Sequence Model. 
        Args:
            encoder (Callable): Encoder Unit
            decoder (Callable): Decoder Unit
        """
        super(SeqToSeq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        if self.encoder.hidden_size != self.decoder.hidden_size:
            raise Exception("Hidden Size mismatch")
        
    def forward(self, inputs: torch.tensor, targets: torch.tensor, teacher_forcing_ratio: float = 0.5):
        """Forward Propagation

        Args:
            inputs (torch.tensor): Inputs sequence to encoder. Dimension: (bs, seq_len)
            targets (torch.tensor): Target sequence to decoder. Dimension: 
            teacher_forcing_ratio (float, optional): _description_. Defaults to 0.0.
        """
        ## Inputs  (bs, enc_seq_len)
        ## Targets (bs, target_seq_len)
        batch_size, target_len = targets.shape
        target_vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(config.DEVICE) # (bs, target_len, target_vocab_size)
        encoder_hiddens, encoder_final_hidden = self.encoder(inputs) # (bs, enc_seq_len, hidden_size), (num_layers*num_directions, bs, hidden_size)
        decoder_next_input = targets[:,0] # (bs)
        
        for time in range(1, target_len):
            output, encoder_final_hidden = self.decoder(decoder_next_input, encoder_hiddens, encoder_final_hidden) # (bs, dec_vocab_size), (bs, no_of_directions * num_layers, hidden_size)
            outputs[:,time,:] = output # (bs, dec_vocab_size)
            teacher_force = torch.random()
            pred = output.argmax(1) # (bs)
            decoder_next_input = pred if teacher_force > teacher_forcing_ratio else targets[:,time] # (bs)

        return outputs
        
            
if __name__ == "__main__":
    from src.models.rnn.rnn import RNNDecoder, RNNEncoder
    encoder = RNNEncoder(
        vocab_size = 5000, emb_size = 300,
        hidden_size = 64, rnn_type = "LSTM"
        )
    
    decoder = RNNDecoder(
        vocab_size = 3000, emb_size = 300,
        hidden_size = encoder.hidden_size, rnn_type = encoder.rnn_type,
        num_layers = encoder.num_layers, attention = True
        )
    
    seq2seq = SeqToSeq(encoder, decoder)
    
    sample_inputs = torch.randint(0, 4999, size = (5,20))
    sample_targets = torch.randint(0, 3000, size = (5,15))
    
    out = seq2seq(sample_inputs, sample_targets)
    print(out.size())