import torch
import torch.nn as nn

from ...nn.utils import get_sequence_lengths


__all__ = ["RNNEncoder", "LSTMEncoder", "GRUEncoder"]


class _RNNBaseEncoder(nn.Module):
    def __init__(self, module, return_sequences):
        super(_RNNBaseEncoder, self).__init__()
        self.module = module
        self.return_sequences = return_sequences

    def get_input_dim(self) -> int:
        return self.module.input_size

    def get_output_dim(self) -> int:
        return self.module.hidden_size * (
            2 if self.module.bidirectional else 1
        )

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.BoolTensor = None,
        hidden_state: torch.Tensor = None,
    ) -> torch.Tensor:
        if mask is None:
            outputs, _ = self.module(inputs, hidden_state)
            if self.return_sequences:
                return outputs
            else:
                return outputs[:, -1, :]

        total_length = inputs.size(1)
        lengths = get_sequence_lengths(mask)
        packed_inputs = nn.utils.rnn.pack_padded_sequence(
            inputs, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, state = self.module(packed_inputs, hidden_state)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs, batch_first=True, total_length=total_length
        )
        if self.return_sequences:
            return outputs
        else:
            if isinstance(state, tuple):
                state = state[0]
            state = state.transpose(0, 1)
            num_directions = 2 if self.module.bidirectional else 1
            last_state = state[:, -num_directions:, :]
            return last_state.contiguous().view([-1, self.get_output_dim()])


class RNNEncoder(_RNNBaseEncoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        return_sequences: bool = False,
    ):
        module = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        super().__init__(module, return_sequences)


class LSTMEncoder(_RNNBaseEncoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        return_sequences: bool = False,
    ):
        module = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        super().__init__(module, return_sequences)


class GRUEncoder(_RNNBaseEncoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        return_sequences: bool = False,
    ):
        module = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        super().__init__(module, return_sequences)
