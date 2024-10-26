import torch
import torch.nn as nn

from src.models.transformers.block import get_masked_attention_mask

def greedy_decode(
    model: nn.Module,
    src: torch.tensor, # Single sequence
    max_tokens: int = 512,
    cls_token_id: int = 101,
    eos_token_id: int = 102,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    if src.ndim == 1:
        src = src.unsqueeze(0)
        
    src_mask = torch.ones(size=(1, 1, src.size(-1)), device=device)
    encoder_output = model.encode(src, src_mask)

    tgt = torch.tensor([cls_token_id], dtype=torch.long, device=device) # [seq_len]

    for _ in range(max_tokens):
        tgt_mask = get_masked_attention_mask(tgt.size(-1)).to(device)
        last_probs = model.decode(
            tgt=tgt.unsqueeze(0),
            memory=encoder_output,
            src_mask=src_mask,
            tgt_mask=tgt_mask)[:, -1, :] # (1, tgt_vocab_size)

        next_token_id = last_probs.argmax(dim=-1)[0] # (1, 1)
        tgt = torch.cat([tgt, next_token_id], dim=-1)
        if next_token_id == eos_token_id:
            break
        
    return tgt

