import torch
import torch.nn as nn

class CustomCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        label_smoothing: float=0.1,
        ignore_index: int=0,
        use_kl_divergence: bool=False
    ):
        super(CustomCrossEntropyLoss, self).__init__()
        self.use_kl_divergence = use_kl_divergence
        if use_kl_divergence:
            self.criterion = nn.KLDivLoss(reduction="sum")
            self.ignore_index = ignore_index
            self.confidence = 1 - label_smoothing
            self.smoothing = label_smoothing
            self.true_dist = None

        else:
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
                ignore_index=0
            )

    def forward(
        self, preds, labels
    ):
        if self.use_kl_divergence:
            num_classes = preds.size(1)
            true_dist = preds.data.clone()
            negative_prob = self.smoothing / (num_classes - 2) # Exclude <PAD> and <CLS> tokens 
            true_dist.fill_(negative_prob) # Update probability for negative classes
            true_dist.scatter_(1, labels.data.unsqueeze(1), self.confidence) # Update probability for positive class
            true_dist[:, self.ignore_index] = 0
            mask = torch.nonzero(labels.data == self.ignore_index)
            if mask.dim() > 0:
                true_dist.index_fill_(0, mask.squeeze(), 0.0)
            self.true_dist = true_dist
            return self.criterion(
                preds, true_dist.clone().detach()
            )
        else:
            return self.criterion(preds, labels)