# -*- coding: utf-8 -*-

from typing import Dict

import tensorflow as tf

from .. import utils
from ...data import Field


def DSSM(features: Dict[str, Field],
         targets: Dict[str, Field],
         text_embedder,
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

    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100))
    encoded_premise = lstm(embedded_premise)
    encoded_hypothesis = lstm(embedded_hypothesis)

    mlp = utils.build_mlp(mlp_num_layers, mlp_num_units, mlp_num_fan_out,
                          mlp_activation)
    encoded_premise = mlp(encoded_premise)
    encoded_hypothesis = mlp(encoded_hypothesis)
    sim = tf.keras.layers.Dot(axes=[1, 1], normalize=True)(
        [encoded_premise, encoded_hypothesis])
    preds = tf.keras.layers.Dense(len(targets[label_field].vocab),
                                  activation='softmax',
                                  name=label_field)(sim)
    return tf.keras.models.Model(inputs=list(inputs.values()),
                                 outputs=preds,
                                 name='DSSM')
