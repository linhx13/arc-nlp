from typing import Union, List, Dict

import tensorflow as tf


class Transform:

    def __init__(self):
        pass

    def preprocess(self, x):
        return x

    def encode(self):
        raise NotImplementedError

    def postprocessing(self, batch):
        return batch

    def padded_shape(self):
        raise NotImplementedError


class TextFeature(Transform):

    def __init__(self, tokenizer, max_len=None):
        super(TextFeature, self).__init__()
        self.tokenizer = tokenizer
        self.vocab = None
        self.max_len = max_len

    def preprocess(self, x):
        if isinstance(x, tf.Tensor):
            x = x.numpy()
        if isinstance(x, list):
            x = [tf.compat.as_text(t) for t in x]
        else:
            x = tf.compat.as_text(x)
            x = self.tokenizer(x)
        return x

    def encode(self, x):
        if isinstance(x, tf.Tensor):
            x = x.numpy()
        return self.vocab(x)

    def padded_shape(self):
        return [self.max_len]

    def output_type(self):
        return tf.int32


class Label(Transform):

    def __init__(self, sparse_target: bool=False):
        super(Label, self).__init__()
        self.vocab = None
        self.sparse_target = sparse_target

    def preprocess(self, x):
        if isinstance(x, tf.Tensor):
            x = x.numpy()
        if isinstance(x, (str, bytes)):
            x = tf.compat.as_text(x)
        return x

    def encode(self, x):
        if isinstance(x, tf.Tensor):
            x = x.numpy()
        x = self.vocab(x)
        return x

    def postprocessing(self, batch):
        if self.sparse_target:
            return tf.expand_dims(batch, axis=-1)
        else:
            return tf.one_hot(batch, len(self.vocab))

    def padded_shape(self):
        return []

    def output_type(self):
        return tf.int32
