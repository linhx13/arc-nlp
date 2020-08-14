import tensorflow as tf
import numpy as np


def vocab_func(vocab):
    def func(tok_iter):
        return [vocab[tok] for tok in tok_iter]

    return func


def totensor(dtype):
    def func(ids_list):
        return tf.convert_to_tensor(ids_list, dtype=dtype)

    return func


def sequential_transforms(*transforms):
    def func(text_input):
        for transform in transforms:
            text_input = transform(text_input)
        return text_input

    return func


def label_func(vocab):
    def func(x):
        x = vocab[x]
        return np.eye(len(vocab), dtype='int32')[x]

    return func
