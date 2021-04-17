import torch
import torch.nn as nn

from ..utils import get_sequence_lengths


class BOWEncoder(nn.Module):
    def __init__(self, embedding_dim: int, averaged: bool = False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.averaged = averaged

    def get_input_dim(self) -> int:
        return self.embedding_dim

    def get_output_dim(self) -> int:
        return self.embedding_dim

    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor = None):
        # inputs: (batch_size, num_tokens, embedding_dim)
        if mask is not None:
            inputs = inputs * mask.unsqueeze(-1)

        summed = inputs.sum(1)

        if self.averaged:
            if mask is not None:
                lengths = get_sequence_lengths(mask)
                length_mask = lengths > 0
                lengths = torch.max(lengths, lengths.new_ones(1))
            else:
                lengths = inputs.new_full((1,), fill_value=inputs.size(1))
                length_mask = None

            summed = summed / lengths.unsqueeze(-1).float()

            if length_mask is not None:
                summed = summed * (length_mask > 0).unsqueeze(-1)

        return summed
