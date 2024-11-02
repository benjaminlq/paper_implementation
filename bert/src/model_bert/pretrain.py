import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from model_bert.backbone import BERTBackbone

class MaskLanguageHead(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        vocab_size: int = 30522
    ):
        super(MaskLanguageHead, self).__init__()
        self.d_model = d_model
        self.proj = nn.Linear(
            d_model, vocab_size
        )

    def forward(
        self,
        inputs: torch.Tensor
    ):
        return F.log_softmax(self.proj(inputs), dim=-1)

class NextSentencePredictionHead(nn.Module):
    def __init__(
        self,
        d_model: int = 768
    ):
        super(NextSentencePredictionHead, self).__init__()
        self.proj = nn.Linear(d_model, 2)

    def forward(
        self,
        inputs: torch.Tensor
    ):
        cls_embs = inputs[:, 0, :] # (batch_size, d_model) -> (batch_size, 2)
        return F.log_softmax(self.proj(cls_embs), dim=-1)

class BERTForPretraining(nn.Module):
    def __init__(
        self,
        bert: BERTBackbone,
        vocab_size: Optional[int] = None
    ):
        super(BERTForPretraining, self).__init__()
        self.bert = bert
        self.vocab_size = vocab_size or self.bert.vocab_size
        self.d_model = self.bert.d_model
        self.mlm = MaskLanguageHead(
            self.d_model, self.vocab_size
        )
        self.nsp = NextSentencePredictionHead(
            self.d_model
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        **kwargs,
    ):
        final_hiddens = self.bert(
            inputs = input_ids,
            token_type_ids = token_type_ids,
            attn_mask = attn_mask,
            is_causal = is_causal
        )
        return (self.mlm(final_hiddens), self.nsp(final_hiddens))