import torch
import torch.nn.functional as F


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


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.BoolTensor,
    dim: int,
    keepdim: bool = False,
) -> torch.Tensor:
    replaced_tensor = tensor.masked_fill(~mask, 0.0)
    value_sum = torch.sum(replaced_tensor, dim=dim, keepdim=keepdim)
    value_count = torch.sum(mask, dim=dim, keepdim=keepdim)
    return value_sum / value_count.float().clamp(
        min=tiny_value_of_dtype(torch.float)
    )


def masked_softmax(
    tensor: torch.Tensor, mask: torch.BoolTensor, dim: int = -1
) -> torch.Tensor:
    if mask is None:
        return F.softmax(tensor, dim=dim)
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    masked_tensor = tensor.masked_fill(~mask, min_value_of_dtype(tensor.dtype))
    return F.softmax(masked_tensor, dim=dim)


def tiny_value_of_dtype(dtype: torch.dtype):
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))


def info_value_of_dtype(dtype: torch.dtype):
    if dtype == torch.bool:
        raise TypeError("Does not support torch.bool")
    elif dtype.is_floating_point:
        return torch.finfo(dtype)
    else:
        return torch.iinfo(dtype)


def min_value_of_dtype(dtype: torch.dtype):
    return info_value_of_dtype(dtype).min


def max_value_of_dtype(dtype: torch.dtype):
    return info_value_of_dtype(dtype).max
