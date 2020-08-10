# -*- coding: utf-8 -*-

from typing import Dict

import tensorflow as tf

from .. import utils
from ...data import Feature


def CDSSM(features: Dict[str, Feature],
          targets: Dict[str, Feature],
          text_embedder,
          conv_filters: int = 32,
          conv_kernel_size: int = 3,
          conv_activation='relu',
          dropout: float = 0.3,
          mlp_num_layers: int = 3,
          mlp_num_units: int = 300,
          mlp_num_fan_out: int = 128,
          mlp_activation='relu',
          label_field: str = 'label'):
    inputs = utils.create_inputs(features)
    input_premise = inputs['premise']
    input_hypothesis = inputs['hypothesis']
    embedded_premise = text_embedder(input_premise)
    embedded_hypothesis = text_embedder(input_hypothesis)
    encoder = _build_encoder(conv_filters, conv_kernel_size,
                             conv_activation, dropout,
                             mlp_num_layers, mlp_num_units,
                             mlp_num_fan_out, mlp_activation)
    encoded_premise = encoder(embedded_premise)
    encoded_hypothesis = encoder(embedded_hypothesis)
    sim = tf.keras.layers.Dot(axes=[1, 1], normalize=True)(
        [encoded_premise, encoded_hypothesis])
    preds = tf.keras.layers.Dense(len(targets[label_field].vocab),
                                  activation='softmax',
                                  name=label_field)(sim)
    return tf.keras.models.Model(inputs=list(inputs.values()),
                                 outputs=preds,
                                 name="CDSSM")


def _build_encoder(conv_filters, conv_kernel_size, conv_activation, dropout,
                   mlp_num_layers, mlp_num_units, mlp_num_fan_out, mlp_activation):

    conv_layer = tf.keras.layers.Conv1D(
        filters=conv_filters,
        kernel_size=conv_kernel_size,
        activation=conv_activation)
    if dropout:
        dropout_layer = tf.keras.layers.Dropout(dropout)
    pooling = tf.keras.layers.GlobalMaxPooling1D()
    mlp = utils.build_mlp(mlp_num_layers, mlp_num_units,
                          mlp_num_fan_out, mlp_activation)

    def _encoder(x):
        x = conv_layer(x)
        if dropout:
            x = dropout_layer(x)
        x = pooling(x)
        x = mlp(x)
        return x

    return _encoder
