import torch

def get_text_mask(tokens, padding_idx=0) -> torch.BoolTensor:
    if padding_idx is not None:
        return tokens != padding_idx
    else:
        return None
