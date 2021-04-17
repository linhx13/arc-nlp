import torch
import torch.nn as nn

from .utils import get_sequence_lengths

__all__ = ["MaskedRNN", "MaskedLSTM", "MaskedGRU"]


class _MaskedBase(nn.Module):
    def __init__(self, module: nn.RNNBase):
        super(_MaskedBase, self).__init__()
        self.module = module

    @property
    def input_size(self):
        return self.module.input_size

    @property
    def hidden_size(self):
        return self.module.hidden_size

    @property
    def num_layers(self):
        return self.module.num_layers

    @property
    def bias(self):
        return self.module.bias

    @property
    def batch_first(self):
        return self.module.batch_first

    @property
    def dropout(self):
        return self.module.dropout

    @property
    def bidirectional(self):
        return self.module.bidirectional

    def forward(
        self,
        inputs: torch.Tensor,
        hx: torch.Tensor = None,
        mask: torch.BoolTensor = None,
    ) -> torch.Tensor:
        if self.batch_first:
            batch_size, total_length, _ = inputs.size()
        else:
            total_length, batch_size, _ = inputs.size()
        if mask is not None and not isinstance(
            inputs, nn.utils.rnn.PackedSequence
        ):
            lengths = get_sequence_lengths(mask)
            sort_lengths, sort_idx = torch.sort(
                lengths, dim=0, descending=True
            )
            if self.batch_first:
                inputs = inputs[sort_idx]
            else:
                inputs = inputs[:, sort_idx]
            inputs = nn.utils.rnn.pack_padded_sequence(
                inputs, sort_lengths.cpu(), batch_first=self.batch_first
            )
            outputs, hx = self.module(inputs, hx)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs,
                batch_first=self.batch_first,
                total_length=total_length,
            )
            _, unsort_idx = torch.sort(sort_idx, dim=0, descending=False)
            if self.batch_first:
                outputs = outputs[unsort_idx]
            else:
                outputs = outputs[:, unsort_idx]
            if isinstance(hx, tuple):
                hx = hx[0][:, unsort_idx], hx[1][:, unsort_idx]
            else:
                hx = hx[:, unsort_idx]
        else:
            outputs, hx = self.module(inputs, hx)
        return outputs, hx


class MaskedRNN(_MaskedBase):
    def __init__(self, *args, **kwargs):
        module = nn.RNN(*args, **kwargs)
        super(MaskedRNN, self).__init__(module)


class MaskedLSTM(_MaskedBase):
    def __init__(self, *args, **kwargs):
        module = nn.LSTM(*args, **kwargs)
        super(MaskedLSTM, self).__init__(module)


class MaskedGRU(_MaskedBase):
    def __init__(self, *args, **kwargs):
        module = nn.GRU(*args, **kwargs)
        super(MaskedGRU, self).__init__(module)
