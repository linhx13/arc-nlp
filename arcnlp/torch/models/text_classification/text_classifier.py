import torch
import torch.nn as nn

from ...nn.utils import get_tokens_mask


class TextClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedder: nn.Module,
        seq2vec_encoder: nn.Module,
        seq2seq_encoder: nn.Module = None,
        dropout: float = 0.1,
        padding_idx=0,
    ):
        super().__init__()
        self.embedder = embedder
        self.seq2vec_encoder = seq2vec_encoder
        self.seq2seq_encoder = seq2seq_encoder
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.fc = nn.Linear(self.seq2vec_encoder.get_output_dim(), num_classes)
        self.padding_idx = padding_idx

    def forward(self, tokens: torch.Tensor):
        mask = get_tokens_mask(tokens, self.padding_idx)
        embedded = self.embedder(tokens)
        if self.seq2seq_encoder:
            embedded = self.seq2seq_encoder(embedded, mask)
        encoded = self.seq2vec_encoder(embedded, mask)
        if self.dropout:
            encoded = self.dropout(encoded)
        return self.fc(encoded)
