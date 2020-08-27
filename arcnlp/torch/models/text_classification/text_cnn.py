import torch
from torch import nn

from ...modules import CnnEncoder
from ...nn.utils import get_text_mask


class TextCNN(nn.Module):

    def __init__(self, num_classes, token_embedder,
                 filters=100, kernel_sizes=(2, 3, 4, 5),
                 conv_layer_activation=None, dropout=0.1):
        super(TextCNN, self).__init__()
        self.token_embedder = token_embedder
        self.encoder = CnnEncoder(token_embedder.embedding_dim, filters=filters,
                                  kernel_sizes=kernel_sizes,
                                  conv_layer_activation=conv_layer_activation)
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.fc = nn.Linear(self.encoder.get_output_dim(), num_classes)

    def forward(self, tokens):
        embedded = self.token_embedder(tokens)
        mask = get_text_mask(tokens, self.token_embedder.padding_idx)
        encoded = self.encoder(embedded, mask)
        if self.dropout:
            encoded = self.dropout(encoded)
        return self.fc(encoded)
