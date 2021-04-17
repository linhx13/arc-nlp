import torch
import torch.nn as nn

from ...nn import RNNEncoder, LSTMEncoder, GRUEncoder
from ...nn.utils import get_tokens_mask


class TextRNN(nn.Module):
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
        super(TextRNN, self).__init__()
        self.embedder = embedder
        if rnn_type == "rnn":
            encoder_cls = RNN
        elif rnn_type == "lstm":
            encoder_cls = LSTMEncoder
        elif rnn_type == "gru":
            encoder_cls = GRUEncoder
        else:
            raise ValueError("rnn_type %s is invalid" % rnn_type)
        self.encoder = encoder_cls(
            embedder.embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.padding_idx = padding_idx

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = get_tokens_mask(tokens, padding_idx=self.padding_idx)
        embedded = self.embedder(tokens)
        encoded = self.encoder(embedded, mask=mask)
        if self.dropout:
            encoded = self.dropout(encoded)
        return self.fc(encoded)
