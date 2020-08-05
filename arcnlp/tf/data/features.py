from typing import Union, List, Dict

import tensorflow as tf


class TextFeature:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab = None

    def __call__(self, x) -> List[Union[str, int]]:
        if isinstance(x, tf.Tensor):
            x = x.numpy()
        if isinstance(x, list):
            x = [tf.compat.as_text(t) for t in x]
        elif isinstance(x, (str, bytes)):
            x = tf.compat.as_text(x)
            x = self.tokenizer(x)
        if self.vocab:
            x = self.vocab(x)
        return x


class Label:

    def __init__(self):
        self.vocab = None

    def __call__(self, x) -> Union[str, int]:
        if isinstance(x, tf.Tensor):
            x = x.numpy()
        if isinstance(x, (str, bytes)):
            x = tf.compat.as_text(x)
        if isinstance(x, str) and self.vocab is not None:
            x = self.vocab(x)
        return x
