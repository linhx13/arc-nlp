import torch
import torch.nn as nn

from ...nn.utils import get_tokens_mask, get_rnn_encoder, masked_softmax


class TextRNNAttn(nn.Module):
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
        super(TextRNNAttn, self).__init__()
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
        self.w = nn.Parameter(torch.zeros(self.encoder.output_dim))
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.fc = nn.Linear(self.encoder.output_dim, num_classes)
        self.padding_idx = padding_idx

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = get_tokens_mask(tokens, padding_idx=self.padding_idx)
        embedded = self.embedder(tokens)
        if self.dropout:
            embedded = self.dropout(embedded)
        encoded = self.encoder(embedded, mask=mask)  # [N, L, C]
        if self.dropout:
            encoded = self.dropout(encoded)
        M = F.tanh(encoded)
        alpha = masked_softmax(
            torch.matmul(M, self.w), mask=mask, dim=1
        ).unsqueeze(
            -1
        )  # [N, L, 1]
        out = encoded * alpha  # [N, L, C]
        out = torch.sum(out, dim=1)  # [N, C]
        out = torch.tanh(out)
        if self.dropout:
            out = self.dropout(out)
        out = self.fc(out)
        return out
