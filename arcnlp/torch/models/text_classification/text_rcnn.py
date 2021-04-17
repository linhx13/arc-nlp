import torch
import torch.nn as nn
import torch.nn.functional as F

from ...nn.utils import get_tokens_mask, get_rnn_encoder


class TextRCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedder: nn.Module,
        rnn_type: str = "lstm",
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
        padding_idx: int = 0,
    ):
        super(TextRCNN, self).__init__()
        self.embedder = embedder
        self.encoder = get_rnn_encoder(
            rnn_type,
            input_size=embedder.embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            return_sequences=True,
        )
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.fc = nn.Linear(
            embedder.embedding_dim + self.encoder.output_dim, num_classes
        )
        self.padding_idx = padding_idx

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = get_tokens_mask(tokens, padding_idx=self.padding_idx)
        embedded = self.embedder(tokens)
        encoded = self.encoder(embedded, mask=mask)
        out = torch.cat((embedded, encoded), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = F.max_pool1d(out, out.size(2)).squeeze()
        if self.dropout:
            out = self.dropout(out)
        return self.fc(out)
