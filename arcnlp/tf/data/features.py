from typing import Union, List, Dict

import numpy as np
import tensorflow as tf

from ..constants import PAD_TOKEN


class Feature:

    def __init__(self):
        pass

    def count_vocab(self, x, counter) -> None:
        pass

    def __call__(self, x):
        raise NotImplementedError

    def padded_shape(self):
        raise NotImplementedError

    def output_type(self):
        raise NotImplementedError


class TextFeature(Feature):

    def __init__(self, tokenizer, max_len=None, lower=False,
                 bos_token=None, eos_token=None, pad_token=PAD_TOKEN,
                 pad_first=False, truncate_first=False):
        super(TextFeature, self).__init__()
        self.tokenizer = tokenizer
        self.vocab = None
        self.max_len = max_len
        self.lower = lower

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

    def count_vocab(self, x, counter) -> None:
        tokens = self._tokenize(x)
        counter.update(tokens)

    def __call__(self, x) -> np.array:
        tokens = self._tokenize(x)
        return np.array(self.vocab(tokens), dtype='int32')

    def padded_shape(self):
        return [self.max_len]

    def output_type(self):
        return tf.int32


class Label(Feature):

    def __init__(self, sparse_target: bool=False):
        super(Label, self).__init__()
        self.vocab = None
        self.sparse_target = sparse_target

    def _as_text(self, x):
        if isinstance(x, tf.Tensor):
            x = x.numpy()
        if isinstance(x, (str, bytes)):
            x = tf.compat.as_text(x)
        return x

    def count_vocab(self, x, counter) -> None:
        token = self._as_text(x)
        counter[token] += 1

    def __call__(self, x) -> np.array:
        idx = self.vocab(self._as_text(x))
        if self.sparse_target:
            return np.expand_dims(idx, axis=-1)
        else:
            return np.eye(len(self.vocab), dtype='int32')[idx]

    def padded_shape(self):
        return [len(self.vocab)]

    def output_type(self):
        return tf.int32
