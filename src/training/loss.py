import torch
from torch import nn


class MNRloss(nn.Module):
    def __init__(self, label_smoothing=0):
        super().__init__()
        self.loss_f = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, sentence_embedding_A: torch.Tensor, sentence_embedding_B: torch.Tensor):
        # Compute similarity matrix
        scores = torch.mm(sentence_embedding_A, sentence_embedding_B.transpose(0, 1))
        # Compute labels
        labels = torch.arange(len(scores), dtype=torch.long, device=scores.device)
        return self.loss_f(scores, labels)