import torch
import torch.nn as nn
import torch.nn.functional as F

from ...nn.utils import get_tokens_mask, masked_mean


class FastText(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedder: nn.Module,
        dropout: float = 0.1,
        padding_idx: int = 0,
    ):
        super(FastText, self).__init__()
        self.embedder = embedder
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.fc1 = nn.Linear(
            self.embedder.embedding_dim, self.embedder.embedding_dim
        )
        self.fc2 = nn.Linear(self.embedder.embedding_dim, num_classes)
        self.padding_idx = padding_idx

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = get_tokens_mask(tokens, padding_idx=self.padding_idx)
        embedded = self.embedder(tokens)
        out = masked_mean(embedded, mask.unsqueeze(-1), dim=1)
        if self.dropout:
            out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out
