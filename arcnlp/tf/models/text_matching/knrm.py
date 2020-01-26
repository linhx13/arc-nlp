# -*- coding: utf-8 -*-

from typing import Dict

import tensorflow as tf

from .. import utils
from ...data import Field
from ...layers.text_embedders import TextEmbedder


def KNRM(features: Dict[str, Field],
         targets: Dict[str, Field],
         text_embedder: TextEmbedder,
         kernel_num: int = 11,
         sigma: float = 0.1,
         exact_sigma: float = 0.001,
         label_field: str = 'label'):
    inputs = utils.create_inputs(features)
    input_premise = utils.get_text_inputs(inputs, 'premise')
    input_hypothesis = utils.get_text_inputs(inputs, 'hypothesis')
    embedded_premise = text_embedder(input_premise)
    embedded_hypothesis = text_embedder(input_hypothesis)

    mm = tf.keras.layers.Dot(axes=[2, 2], normalize=True)(
        [embedded_premise, embedded_hypothesis])

    km = []
    for i in range(kernel_num):
        _mu = 1. / (kernel_num - 1) + (2. * i) / (kernel_num - 1) - 1.0
        _sigma = sigma
        if _mu > 1.0:
            _sigma = exact_sigma
            _mu = 1.0
        mm_exp = _kernel_layer(_mu, _sigma)(mm)
        mm_doc_sum = tf.keras.layers.Lambda(
            lambda x: tf.reduce_sum(x, 2))(mm_exp)
        mm_log = tf.keras.layers.Activation(tf.math.log1p)(mm_doc_sum)
        mm_sum = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, 1))(mm_log)
        km.append(mm_sum)

    phi = tf.keras.layers.Lambda(lambda x: tf.stack(x, 1))(km)
    preds = tf.keras.layers.Dense(len(targets[label_field].vocab),
                                  activation='softmax',
                                  name=label_field)(phi)
    return tf.keras.models.Model(inputs=list(inputs.values()),
                                 outputs=preds,
                                 name='KNRM')


def _kernel_layer(mu, sigma):

    def kernel(x):
        return tf.math.exp(-0.5 * (x - mu) * (x - mu) / sigma / sigma)

    return tf.keras.layers.Activation(kernel)
