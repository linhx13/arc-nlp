# -*- coding: utf-8 -*-

from typing import Dict, Iterable

import tensorflow as tf

from .. import utils
from ...data import Field
from ...layers.text_embedders import TextEmbedder
from ...layers import MatchingLayer


def ArcII(features: Dict[str, Field],
          targets: Dict[str, Field],
          text_embedder: TextEmbedder,
          conv_1d_filters: int = 300,
          conv_1d_kernel_size: int = 3,
          conv_1d_activation="relu",
          conv_2d_filters: Iterable = [16, 32],
          conv_2d_kernel_sizes: Iterable = [[3, 3], [3, 3]],
          conv_2d_activation="relu",
          pool_2d_sizes: Iterable = [[2, 2], [2, 2]],
          dropout: float = 0.5,
          label_field: str = 'label'):
    assert len(conv_2d_filters) == len(conv_2d_kernel_sizes)
    assert len(conv_2d_filters) == len(pool_2d_sizes)

    inputs = utils.create_inputs(features)
    input_premise = utils.get_text_inputs(inputs, 'premise')
    input_hypothesis = utils.get_text_inputs(inputs, 'hypothesis')
    embedded_premise = text_embedder(input_premise)
    embedded_hypothesis = text_embedder(input_hypothesis)

    conv_1d_premise = tf.keras.layers.Conv1D(conv_1d_filters,
                                             conv_1d_kernel_size,
                                             activation=conv_1d_activation,
                                             padding='same')(embedded_premise)
    conv_1d_hypothesis = tf.keras.layers.Conv1D(conv_1d_filters,
                                                conv_1d_kernel_size,
                                                activation=conv_1d_activation,
                                                padding='same')(embedded_hypothesis)

    matching_layer = MatchingLayer(matching_type="plus")
    embedded_cross = matching_layer([conv_1d_premise, conv_1d_hypothesis])

    for i in range(len(conv_2d_filters)):
        embedded_cross = _conv_pool_block(embedded_cross,
                                          conv_2d_filters[i],
                                          conv_2d_kernel_sizes[i],
                                          conv_2d_activation,
                                          pool_2d_sizes[i])

    encoded = tf.keras.layers.Flatten()(embedded_cross)
    if dropout:
        encoded = tf.keras.layers.Dropout(dropout)(encoded)
    preds = tf.keras.layers.Dense(len(targets[label_field].vocab),
                                  activation='softmax',
                                  name=label_field)(encoded)
    return tf.keras.models.Model(inputs=list(inputs.values()),
                                 outputs=preds,
                                 name='ArcII')


def _conv_pool_block(inputs,
                     filters,
                     kernel_size,
                     activation,
                     pool_size):
    outputs = tf.keras.layers.Conv2D(filters,
                                     kernel_size,
                                     activation=activation,
                                     padding='same')(inputs)
    outputs = tf.keras.layers.MaxPooling2D(pool_size)(outputs)
    return outputs
