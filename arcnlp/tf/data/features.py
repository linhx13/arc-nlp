from typing import Union, List, Dict

import numpy as np
import tensorflow as tf

from ..constants import PAD_TOKEN


class Feature:

    def __init__(self):
        pass

    def count_vocab(self, x, counter) -> None:
        pass

    def encode(self, x):
        raise NotImplementedError

    def padded_shape(self):
        raise NotImplementedError

    def padding_value(self):
        raise NotImplementedError

    def output_type(self):
        raise NotImplementedError


class TextFeature(Feature):

    def __init__(self, tokenizer=None, lower=False, max_len=None,
                 bos_token=None, eos_token=None, pad_token=PAD_TOKEN,
                 pad_first=False, truncate_first=False):
        super(TextFeature, self).__init__()
        self.vocab = None
        self.tokenizer = tokenizer if tokenizer else str.split()
        self.lower = lower
        self.max_len = max_len
        self.bos_token = bos_token
        self.eos_token = bos_token
        self.pad_token = pad_token
        self.pad_first = pad_first
        self.truncate_first = truncate_first

    def count_vocab(self, x, counter) -> None:
        tokens = self._tokenize(x)
        counter.update(tokens)

    def encode(self, x) -> np.array:
        tokens = self._pad(x)
        return np.array(self.vocab(tokens), dtype='int32')

    def _tokenize(self, x) -> List[str]:
        if isinstance(x, tf.Tensor):
            x = x.numpy()
        if isinstance(x, list):
            x = [tf.compat.as_text(t) for t in x]
        else:
            x = tf.compat.as_text(x)
            x = self.tokenizer(x)
        if self.lower:
            x = [t.lower() for t in x]
        return x

    def _pad(self, x):
        tokens = self._tokenize(x)
        if self.max_len is None:
            return tokens
        max_len = self.max_len + (self.bos_token, self.eos_token).count(None) - 2
        if self.pad_first:
            padded = ([self.pad_token] * max(0, max_len - len(tokens))) \
                + ([] if self.bos_token is None else [self.bos_token]) \
                + (tokens[-max_len:] if self.truncate_first else tokens[:mx_len]) \
                + ([] if self.eos_token is None else [self.eos_token])
        else:
            padded = ([] if self.bos_token is None else [self.bos_token]) \
                + (tokens[-max_len:] if self.truncate_first else tokens[:max_len]) \
                + ([] if self.eos_token is None else [self.eos_token]) \
                + ([self.pad_token] * max(0, max_len - len(tokens)))
        return tokens

    def padded_shape(self):
        return [self.max_len]

    def padding_value(self):
        return self.vocab[self.pad_token]

    def output_type(self):
        return tf.int32


class Label(Feature):

    def __init__(self, sparse_target: bool=False):
        super(Label, self).__init__()
        self.vocab = None
        self.sparse_target = sparse_target

    def count_vocab(self, x, counter) -> None:
        token = self._as_text(x)
        counter[token] += 1

    def encode(self, x) -> np.array:
        idx = self.vocab(self._as_text(x))
        if self.sparse_target:
            return np.expand_dims(idx, axis=-1)
        else:
            return np.eye(len(self.vocab), dtype='int32')[idx]

    def _as_text(self, x):
        if isinstance(x, tf.Tensor):
            x = x.numpy()
        if isinstance(x, (str, bytes)):
            x = tf.compat.as_text(x)
        return x

    def padded_shape(self):
        return [len(self.vocab)]

    def paddding_value(self):
        return 0

    def output_type(self):
        return tf.int32
