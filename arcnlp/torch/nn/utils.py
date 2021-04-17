import torch


def get_tokens_mask(tokens: torch.Tensor, padding_idx=0) -> torch.BoolTensor:
    if padding_idx is not None:
        return tokens != padding_idx
    else:
        return None


def get_sequence_mask(
    lengths: torch.Tensor, max_length: int
) -> torch.BoolTensor:
    ones = lengths.new_ones(lengths.size(0), max_length)
    range_tensor = ones.cumsum(dim=1)
    return lengths.unsqueeze(1) >= range_tensor


def get_sequence_lengths(mask: torch.BoolTensor) -> torch.LongTensor:
    return mask.sum(-1)


def get_rnn_encoder(rnn_type, **kwargs):
    from ..nn import RNNEncoder, LSTMEncoder, GRUEncoder

    mapping = {"rnn": RNNEncoder, "lstm": LSTMEncoder, "gru": GRUEncoder}
    encoder_cls = mapping.get(rnn_type)
    if rnn_type is None:
        raise ValueError(f"rnn_type {rnn_type} is invalid")
    return encoder_cls(**kwargs)
