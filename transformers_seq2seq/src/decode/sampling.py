import torch
import torch.nn as nn

from typing import Optional
from src.models.transformers.block import get_masked_attention_mask

def top_k_filter(
    probs: torch.tensor, # (batch_size, vocab_size)
    top_k: int
) -> torch.tensor:
    _, top_k_indices = torch.topk(dim = -1)
    filtered_indices = top_k_indices[:, top_k:]
    probs[:, filtered_indices] = 0.0
    return probs

def top_p_filter(
    probs: torch.tensor,
    top_p: float = 1.0
):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    cutoff_indices = (cumulative_probs > top_p)
    cutoff_indices = (~cutoff_indices).sum(dim=-1)
    
    # Scatter the sorted logits back to the original ordering
    for row_idx in range(sorted_probs.size(0)):
        sorted_probs[row_idx, cutoff_indices[row_idx]] = 0.0

    # Apply softmax to get probabilities, then sample
    probs = probs.scatter_(1, sorted_indices, sorted_probs)
    return probs
    
def greedy_sampling_decode(
    model: nn.Module,
    src: torch.tensor, # Single sequence
    max_tokens: int = 512,
    cls_token_id: int = 101,
    eos_token_id: int = 102,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    top_p: float = 1.0
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
            tgt_mask=tgt_mask,
            temperature=temperature
            )[:, -1, :] # (batch_size, tgt_vocab_size)

        # Update probability vector here using top_p and top_k
        if 0 < top_p < 1.0:
            last_probs = top_p_filter(last_probs, top_p)
        if 0 < top_k:
            last_probs = top_k_filter(last_probs, top_k)
            
        normalized_probs = last_probs / last_probs.sum(dim=-1, keepdim=True)
        next_token_id = torch.multinomial(normalized_probs, num_samples=1)

        tgt = torch.cat([tgt, next_token_id], dim=-1)
        if next_token_id == eos_token_id:
            break
        
    return tgt
