# -*- coding: utf-8 -*-

from typing import Dict

import tensorflow as tf

# from ..data import Field
from ..data import Feature


def create_inputs(features: Dict[str, Feature]):
    inputs = {n: tf.keras.layers.Input(shape=(f.max_len,), name=n)
              for n, f in features.items()}
    return inputs


def get_text_inputs(inputs: Dict[str, tf.Tensor], name) -> Dict[str, tf.Tensor]:
    res = {}
    for n, t in inputs.items():
        arr = n.split(".", 1)
        if arr[0] == name:
            res[arr[1]] = t
    return res


def build_mlp(num_layers, num_units, num_fan_out, activation):

    def _mlp(x):
        for _ in range(num_layers):
            x = tf.keras.layers.Dense(num_units, activation=activation)(x)
        x = tf.keras.layers.Dense(num_fan_out, activation=activation)(x)
        return x

    return _mlp
