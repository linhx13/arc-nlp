# -*- coding: utf-8 -*-

from typing import Dict

import tensorflow as tf

from .. import utils
from ...data import Field


def MVLSTM(features: Dict[str, Field],
           targets: Dict[str, Field],
           text_embedder,
           lstm_units: int = 50,
           top_k: int = 10,
           mlp_num_layers: int = 2,
           mlp_num_units: int = 20,
           mlp_num_fan_out: int = 10,
           mlp_activation='relu',
           dropout: float = 0.5,
           label_field: str = 'label'):
    inputs = utils.create_inputs(features)
    input_premise = inputs['premise']
    input_hypothesis = inputs['hypothesis']
    embedded_premise = text_embedder(input_premise)
    embedded_hypothesis = text_embedder(input_hypothesis)

    lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_units, return_sequences=True, dropout=dropout))
    encoded_premise = lstm(embedded_premise)
    encoded_hypothesis = lstm(embedded_hypothesis)

    matching_matrix = tf.keras.layers.Dot(
        axes=[2, 2], normalize=True)([encoded_premise, encoded_hypothesis])
    matching_signals = tf.keras.layers.Reshape((-1,))(matching_matrix)
    matching_topk = tf.keras.layers.Lambda(
        lambda x: tf.nn.top_k(x, k=top_k, sorted=True)[0])(matching_signals)
    mlp = utils.build_mlp(mlp_num_layers, mlp_num_units, mlp_num_fan_out,
                          mlp_activation)
    x = mlp(matching_topk)
    if dropout:
        x = tf.keras.layers.Dropout(dropout)(x)
    preds = tf.keras.layers.Dense(len(targets[label_field].vocab),
                                  activation='softmax',
                                  name=label_field)(x)
    return tf.keras.models.Model(inputs=list(inputs.values()),
                                 outputs=preds,
                                 name='MVLSTM')
