import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from model_bert.backbone import BERTBackbone

class BertForTokenClassification(nn.Module):
    def __init__(
        self,
        bert: BERTBackbone,
        num_classes: int,
        projection_layer: Optional[nn.Module] = None
    ):
        super(BertForTokenClassification, self).__init__()
        self.bert = bert
        self.num_classes = num_classes
        self.d_model = self.bert.d_model
        self.proj = projection_layer or nn.Linear(
            self.d_model, num_classes
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ):
        final_hiddens = self.bert(
            inputs = input_ids,
            token_type_ids = token_type_ids,
            attn_mask = attn_mask,
            is_causal = is_causal
        )
        return self.proj(final_hiddens)
    
class BertForSequenceClassification(nn.Module):
    def __init__(
        self,
        bert: BERTBackbone,
        num_classes: int,
        projection_layer: Optional[nn.Module] = None
    ):
        super(BertForSequenceClassification, self).__init__()
        self.bert = bert
        self.num_classes = num_classes
        self.d_model = self.bert.d_model
        self.proj = projection_layer or nn.Linear(
            self.d_model, num_classes
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ):
        final_hiddens = self.bert(
            inputs = input_ids,
            token_type_ids = token_type_ids,
            attn_mask = attn_mask,
            is_causal = is_causal
        )
        return self.proj(final_hiddens[:, 0, :])

class BertForQuestionAnswering(nn.Module):
    def __init__(
        self,
        bert: BERTBackbone,
        num_classes: int,
        projection_layer: Optional[nn.Module] = None
    ):
        super(BertForSequenceClassification, self).__init__()
        self.bert = bert
        self.num_classes = num_classes
        self.d_model = self.bert.d_model
        self.proj = projection_layer or nn.Linear(
            self.d_model, 2 # For start & end
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ):
        final_hiddens = self.bert(
            inputs = input_ids,
            token_type_ids = token_type_ids,
            attn_mask = attn_mask,
            is_causal = is_causal
        )

        logits = self.proj(final_hiddens) # (batch_size, seq_len, 2)
        start_logits = logits[:, :, 0] # batch size, seq_len
        end_logits = logits[:, :, 1] # batch_size, seq_len

        return start_logits, end_logits