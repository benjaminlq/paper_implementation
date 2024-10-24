import torch
import torch.nn as nn

from .component import (
    TransformerEncoder, TransformerDecoder, Generator
)
from .embeddings import TokenEmbeddings, SinusoidalPositionalEncoding
from typing import Optional
from .block import get_masked_attention_mask

class TransformersSeqToSeq(nn.Module):
    def __init__(
        self,
        n: int=6,
        h: int=8,
        d_model: int=512,
        src_vocab_size: int=30000,
        tgt_vocab_size: int=30000,
        share_embeddings: bool=False,
        max_tokens: int=4096
    ):
        super(TransformersSeqToSeq, self).__init__()
        self.n, self.h, self.d_model = n, h, d_model
        self.encoder = TransformerEncoder(n=n, h=h, d_model=d_model)
        self.decoder = TransformerDecoder(n=n, h=h, d_model=d_model)
        self.src_embeddings = TokenEmbeddings(d_model=d_model, vocab_size=src_vocab_size)
        if share_embeddings:
            self.tgt_embeddings = self.src_embeddings
            self.generator = Generator(d_model=d_model, vocab_size=src_vocab_size)
        else:
            self.tgt_embeddings = TokenEmbeddings(d_model=d_model, vocab_size=tgt_vocab_size)
            self.generator = Generator(d_model=d_model, vocab_size=tgt_vocab_size)

        self.positional_encoding = SinusoidalPositionalEncoding(d_model=d_model, max_len=5000)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.max_tokens=max_tokens

    def encode(
        self, src: torch.tensor, src_mask: Optional[torch.tensor] = None
    ):
        # src = (batch_size, src_seq_len)
        src_token_embeddings = self.src_embeddings(src)
        src_embeddings = self.positional_encoding(src_token_embeddings)
        return self.encoder(x=src_embeddings, mask=src_mask)

    def decode(
        self, tgt, memory, src_mask, tgt_mask
    ):
        # tgt = (batch_size, tgt_seq_len)
        # memory = (batch_size, src_seq_len, d_model)
        tgt_token_embeddings = self.tgt_embeddings(tgt)
        tgt_embeddings = self.positional_encoding(tgt_token_embeddings)
        decoder_output = self.decoder(x=tgt_embeddings, encoder_output=memory, src_mask=src_mask, tgt_mask=tgt_mask) # (batch_size, seq_len, d_model)
        probs = self.generator(decoder_output)
        return probs

    def forward(
        self, src: torch.tensor, tgt: torch.tensor, src_mask: Optional[torch.tensor] = None, tgt_mask: Optional[torch.tensor] = None
    ):
        # src = (batch_size, src_seq_len)
        # tgt = (batch_size, tgt_seq_len)
        memory = self.encode(src, src_mask)
        probs = self.decode(tgt=tgt, memory=memory, src_mask=src_mask, tgt_mask=tgt_mask)
        return probs

    def greedy_decode(
        self,
        src: torch.tensor,
        src_mask: Optional[torch.tensor] = None,
        max_tokens: Optional[int] = None,
        cls_token_id: int = 101,
        eos_token_id: int = 102,
    ):
        device = next(self.parameters()).device
        self.eval()
        max_tokens = max_tokens or self.max_tokens

        if src.ndim == 1:
            src = src.unsqueeze(0)

        batch_size = src.size(0)
        encoder_output = self.encode(src, src_mask)

        tgt = torch.ones(size=(batch_size, 1), dtype=torch.long, device=device) * cls_token_id

        current_token = 1
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        while current_token <= max_tokens and (not is_finished.all()):
            tgt_mask = get_masked_attention_mask(tgt.size(-1)).to(device)
            last_probs = self.decode(
                tgt=tgt,
                memory=encoder_output,
                src_mask=src_mask,
                tgt_mask=tgt_mask)[:, -1, :]

            next_token_id = last_probs.argmax(dim=-1)
            tgt = torch.cat([tgt, next_token_id.unsqueeze(-1)], dim=-1)
            is_finished = is_finished | (next_token_id == eos_token_id)
            current_token += 1

        preds = []
        for pred in tgt:
            token_seq = []
            for token_id in pred:
                token_seq.append(token_id.item())
                if token_id.item() == eos_token_id:
                    break
            preds.append(token_seq)
        return preds
    
    def beam_search_decode(
        self,
        src: torch.tensor,
        src_mask: Optional[torch.tensor] = None,
        max_tokens: Optional[int] = None,
        cls_token_id: int = 101, eos_token_id: int = 102,
        beam_width: int = 4
    ):
        device = next(self.parameters()).device
        self.eval()
        max_tokens = max_tokens or self.max_tokens

        if src.ndim == 1:
            src = src.unsqueeze(0)

        encoder_output = self.encode(src, src_mask)

        beams = [(torch.tensor([cls_token_id], device=device), 0)]  # List of tuples (sequence, score)

        for _ in range(max_tokens):
            all_candidates = []
            for seq, score in beams:
                tgt = seq.unsqueeze(0)  # Add batch dimension
                tgt_mask = get_masked_attention_mask(tgt.size(-1)).to(device)
                # Decode the current sequence
                out = self.decode(
                    tgt,
                    encoder_output,
                    src_mask=src_mask,
                    tgt_mask=tgt_mask
                    ) 
                last_probs = out[:, -1, :].squeeze(0)

                # Expand each beam with all possible next tokens
                for i in range(last_probs.size(-1)):
                    candidate = (torch.cat([seq, torch.tensor([i], device=device)]), score + last_probs[i].item())
                    all_candidates.append(candidate)

            # Select the top beam_width candidates
            all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            beams = all_candidates[:beam_width]

            # Check if all beams have reached the end token
            if all(seq[-1] == eos_token_id for seq, _ in beams):
                break

        # Select the best beam
        best_sequence = sorted(beams, key=lambda x: x[1], reverse=True)[0][0]
        return best_sequence