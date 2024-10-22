import torch
import torch.nn as nn
from src.models.transformers.block import LayerNorm
from src.models.transformers.model import TransformersSeqToSeq

def test_layer_normalization():
    batch_size = 3
    seq_len = 10
    d_model = 4
    eps = 1

    inputs = torch.stack([torch.arange(0, seq_len, dtype=torch.float32) for _ in range(d_model)])
    inputs = inputs.transpose(0, 1)

    scaling_factors = nn.Parameter(torch.tensor([1,2,3,4], dtype=torch.float32))
    bias_factors = nn.Parameter(torch.tensor([1,2,3,4], dtype=torch.float32))
    layer_norm = LayerNorm(d_model=d_model, eps=eps)
    layer_norm.scaling_factor = scaling_factors
    layer_norm.bias_factor = bias_factors

    expected_output = torch.stack([torch.tensor([1,2,3,4], dtype=torch.float32) for _ in range(seq_len)])
    output = layer_norm(inputs)
    assert torch.equal(output, expected_output)

def test_teacher_encoder_decoder():
    src_vocab_size = 300
    tgt_vocab_size = 200
    n = 6
    h = 8
    d_model = 512
    src_len = 20
    tgt_len = 30
    batch_size = 4
    max_tokens = 4096
    model = TransformersSeqToSeq(
        n=n, h=h, d_model=d_model,
        src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size,
        max_tokens=max_tokens
        )
    
    src = torch.randint(0, src_len, size=(batch_size, src_len))
    tgt = torch.randint(0, tgt_len, size=(batch_size, tgt_len))
    outputs = model(src, tgt)
    assert outputs.size() == torch.Size([batch_size, tgt_len, tgt_vocab_size])