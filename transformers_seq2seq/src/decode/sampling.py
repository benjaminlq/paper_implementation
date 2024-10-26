import torch
import torch.nn as nn

from src.models.transformers.block import get_masked_attention_mask

def nucleus_sampling_decode(
    model: nn.Module,
    src: torch.tensor,
    max_tokens: int = 512,
    cls_token_id: int = 101,
    eos_token_id: int = 102,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    if src.ndim == 1:
        src = src.unsqueeze(0)
        
    src_mask = torch.ones(size=(1, 1, src.size(-1)), device=device)
    encoder_output = model.encode(src, src_mask)

    beams = [(torch.tensor([cls_token_id], device=device), 0)]  # List of tuples (sequence, score)

    for _ in range(max_tokens):
        all_candidates = []
        for seq, score in beams:
            tgt = seq.unsqueeze(0)  # Add batch dimension
            tgt_mask = get_masked_attention_mask(tgt.size(-1)).to(device)
            # Decode the current sequence
            out = model.decode(
                tgt,
                encoder_output,
                src_mask=src_mask,
                tgt_mask=tgt_mask
                ) 
            last_probs = out[:, -1, :].squeeze(0) # (tgt_vocab_size, )

    return best_sequence