# -*- coding: utf-8 -*-

from typing import Dict

import tensorflow as tf

from .. import utils
from ...data import Feature


def ArcI(features: Dict[str, Feature],
         targets: Dict[str, Feature],
         text_embedder,
         conv_pool_blocks: int = 2,
         conv_filters: int = 300,
         conv_kernel_size: int = 3,
         conv_activation="relu",
         pool_size: int = 2,
         mlp_num_layers: int = 2,
         mlp_num_units: int = 64,
         mlp_num_fan_out: int = 32,
         mlp_activation='relu',
         dropout: float = 0.5,
         label_field: str = "label"):
    inputs = utils.create_inputs(features)
    input_premise = inputs['premise']
    input_hypothesis = inputs['hypothesis']
    embedded_premise = text_embedder(input_premise)
    embedded_hypothesis = text_embedder(input_hypothesis)

    for _ in range(conv_pool_blocks):
        embedded_premise = _conv_pool_block(embedded_premise,
                                            conv_filters,
                                            conv_kernel_size,
                                            conv_activation,
                                            pool_size)
        embedded_hypothesis = _conv_pool_block(embedded_hypothesis,
                                               conv_filters,
                                               conv_kernel_size,
                                               conv_activation,
                                               pool_size)
    encoded_premise = tf.keras.layers.Flatten()(embedded_premise)
    encoded_hypothesis = tf.keras.layers.Flatten()(embedded_hypothesis)
    encoded = tf.keras.layers.Concatenate()(
        [encoded_premise, encoded_hypothesis])
    if dropout:
        encoded = tf.keras.layers.Dropout(dropout)(encoded)
    mlp = utils.build_mlp(mlp_num_layers, mlp_num_units, mlp_num_fan_out,
                          mlp_activation)
    encoded = mlp(encoded)
    preds = tf.keras.layers.Dense(len(targets[label_field].vocab),
                                  activation='softmax',
                                  name=label_field)(encoded)
    return tf.keras.models.Model(inputs=list(inputs.values()),
                                 outputs=preds,
                                 name='ArcI')


def _conv_pool_block(inputs,
                     filters,
                     kernel_size,
                     activation,
                     pool_size):
    outputs = tf.keras.layers.Conv1D(filters,
                                     kernel_size,
                                     padding='same',
                                     activation=activation)(inputs)
    outputs = tf.keras.layers.MaxPooling1D(pool_size)(outputs)
    return outputs
