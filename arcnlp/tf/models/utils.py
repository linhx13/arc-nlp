# -*- coding: utf-8 -*-

from typing import Dict

import tensorflow as tf

from ..data import Field


def create_inputs(features: Dict[str, Field]):
    inputs = {n: tf.keras.layers.Input(shape=(f.fix_length,), name=n)
              for n, f in features.items}
    return inputs


def get_text_inputs(inputs: Dict[str, tf.Tensor], name) -> Dict[str, tf.Tensor]:
    res = {}
    for n, t in inputs.items():
        arr = n.split(".", 1)
        if arr[0] == name:
            res[arr[1]] = t
    return res
