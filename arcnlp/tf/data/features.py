from typing import Union, List, Dict

import tensorflow as tf


class Feature:

    def __init__(self):
        pass

    def tokenize(self, x) -> List[str]:
        raise NotImplementedError

    def __call__(self, x):
        raise NotImplementedError

    def postprocessing(self, batch):
        return batch

    def padded_shape(self):
        raise NotImplementedError

    def output_type(self):
        raise NotImplementedError


class TextFeature(Feature):

    def __init__(self, tokenizer, max_len=None, lower=False):
        super(TextFeature, self).__init__()
        self.tokenizer = tokenizer
        self.vocab = None
        self.max_len = max_len
        self.lower = lower

    def tokenize(self, x) -> List[str]:
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

    def __call__(self, x) -> List[int]:
        x = self.tokenize(x)
        return self.vocab(x)

    def padded_shape(self):
        return [self.max_len]

    def output_type(self):
        return tf.int32


class Label(Feature):

    def __init__(self, sparse_target: bool=False):
        super(Label, self).__init__()
        self.vocab = None
        self.sparse_target = sparse_target

    def tokenize(self, x) -> List[str]:
        if isinstance(x, tf.Tensor):
            x = x.numpy()
        if isinstance(x, (str, bytes)):
            x = tf.compat.as_text(x)
        return [x]

    def __call__(self, x) -> int:
        x = self.tokenize(x)
        return self.vocab(x[0])

    def postprocessing(self, batch):
        if self.sparse_target:
            return tf.expand_dims(batch, axis=-1)
        else:
            return tf.one_hot(batch, len(self.vocab))

    def padded_shape(self):
        return []

    def output_type(self):
        return tf.int32
