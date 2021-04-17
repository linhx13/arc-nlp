import torch
from torch import nn

from ...nn import CNNEncoder
from ...nn.utils import get_tokens_mask


class TextCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedder: nn.Module,
        num_filters=100,
        kernel_sizes=(1, 3, 5),
        conv_layer_activation=None,
        dropout=0.1,
        padding_idx=0,
    ):
        super(TextCNN, self).__init__()
        self.embedder = embedder
        self.encoder = CNNEncoder(
            input_dim=embedder.embedding_dim,
            num_filters=num_filters,
            kernel_sizes=kernel_sizes,
            conv_layer_activation=conv_layer_activation,
        )
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.padding_idx = 0
        self.fc = nn.Linear(self.encoder.output_dim, num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = get_tokens_mask(tokens, self.padding_idx)
        embedded = self.embedder(tokens)
        encoded = self.encoder(embedded, mask=mask)
        if self.dropout:
            encoded = self.dropout(encoded)
        return self.fc(encoded)
