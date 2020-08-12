# -*- coding: utf-8 -*-

from typing import Dict, Iterable

import tensorflow as tf

from .. import utils
from ...data import Field
from ...layers import MatchingLayer


def MatchPyramid(features: Dict[str, Field],
                 targets: Dict[str, Field],
                 text_embedder,
                 num_blocks: int = 3,
                 conv_filters: Iterable = [8, 16, 32],
                 conv_kernel_sizes: Iterable = [5, 3, 3],
                 conv_activation='relu',
                 pool_sizes: Iterable = [2, 2, 2],
                 dropout: float = 0.5,
                 label_field: str = 'label'):
    assert num_blocks == len(conv_filters) \
        and num_blocks == len(conv_kernel_sizes) \
        and num_blocks == len(pool_sizes)
    inputs = utils.create_inputs(features)
    input_premise = inputs['premise']
    input_hypothesis = inputs['hypothesis']
    embedded_premise = text_embedder(input_premise)
    embedded_hypothesis = text_embedder(input_hypothesis)

    matching_layer = MatchingLayer(matching_type='dot')
    embedded_cross = matching_layer([embedded_premise, embedded_hypothesis])

    for i in range(num_blocks):
        embedded_cross = _conv_pool_block(i,
                                          embedded_cross,
                                          conv_filters[i],
                                          conv_kernel_sizes[i],
                                          conv_activation,
                                          pool_sizes[i])
    flatten = tf.keras.layers.Flatten()(embedded_cross)
    if dropout:
        flatten = tf.keras.layers.Dropout(dropout)(flatten)
    preds = tf.keras.layers.Dense(len(targets[label_field].vocab),
                                  activation='softmax',
                                  name=label_field)(flatten)
    return tf.keras.models.Model(inputs=list(inputs.values()),
                                 outputs=preds,
                                 name='MatchPyramid')


def _conv_pool_block(idx,
                     inputs,
                     filters,
                     kernel_size,
                     activation,
                     pool_size):
    outputs = tf.keras.layers.Conv2D(filters,
                                     kernel_size,
                                     activation=activation,
                                     padding='same',
                                     name='conv_2d_%d' % idx)(inputs)
    outputs = tf.keras.layers.MaxPooling2D(
        pool_size, name='max_pooling_2d_%d' % idx)(outputs)
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    return outputs
