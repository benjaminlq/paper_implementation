import torch

from src.models.transformers.model import TransformersSeqToSeq
from src.models.transformers.component import get_masked_attention_mask

def test_single_generate():
    model_args = dict(
        src_vocab_size = 20, tgt_vocab_size = 10, n = 2
    )
    model = TransformersSeqToSeq(
        **model_args
    )
    model.eval()
    src = torch.arange(3,11,1).unsqueeze(0)
    src_mask = torch.ones(size=(1, 1, src.size(-1)))
    encoder_output = model.encode(src, src_mask)
    max_tokens = 15
    tgt = [1]
    current_token = 1
    while current_token <= max_tokens:
        tgt_seq = torch.LongTensor([tgt])
        tgt_mask = get_masked_attention_mask(tgt_seq.size(-1))
        last_probs = model.decode(
            tgt=tgt_seq,
            memory=encoder_output,
            src_mask=src_mask,
            tgt_mask=tgt_mask)[:, -1, :]
        assert last_probs.size() == torch.Size([1, 10])
        next_token_id = last_probs.argmax(dim=-1).item()
        tgt.append(next_token_id)
        current_token += 1

    assert len(tgt) - 1 == max_tokens 

def test_batch_generate():

    from torch.nn.utils.rnn import pad_sequence

    model_args = dict(
        src_vocab_size = 20, tgt_vocab_size = 10, n = 2
    )

    model = TransformersSeqToSeq(**model_args)
    model.eval()

    batch_size = 5

    src = [torch.arange(3, a) for a in range(7, 7 + batch_size)]

    SRC_PAD_TOKEN_ID = 1
    TGT_CLS_TOKEN_ID = 0
    TGT_EOS_TOKEN_ID = 2

    src = pad_sequence(src, batch_first=True, padding_value=SRC_PAD_TOKEN_ID)
    src_mask = torch.zeros_like(src)
    src_mask = src_mask.masked_fill(src!=SRC_PAD_TOKEN_ID, 1).unsqueeze(-2)
    encoder_output = model.encode(src, src_mask)

    max_tokens = 15

    tgt = torch.ones(size=(batch_size, 1), dtype=torch.long) * TGT_CLS_TOKEN_ID
    current_token = 1

    is_finished = torch.zeros(batch_size, dtype=torch.bool)

    while current_token <= max_tokens and (not is_finished.all()):
        tgt_mask = get_masked_attention_mask(tgt.size(-1))
        last_probs = model.decode(
            tgt=tgt,
            memory=encoder_output,
            src_mask=src_mask,
            tgt_mask=tgt_mask)[:, -1, :]

        next_token_id = last_probs.argmax(dim=-1)
        tgt = torch.cat([tgt, next_token_id.unsqueeze(-1)], dim=-1)
        is_finished = is_finished | (next_token_id == TGT_EOS_TOKEN_ID)
        current_token += 1

    if not is_finished.all():
        assert tgt.size() == torch.Size([batch_size, max_tokens + 1])