# -*- coding: utf-8 -*-

from typing import Dict

import tensorflow as tf

from ... import utils
from ...data import Field
from ...layers.text_embedders import TextEmbedder


def BiLstmMatcher(features: Dict[str, Field],
                  targets: Dict[str, Field],
                  text_embedder: TextEmbedder,
                  lstm_units: int = 128,
                  lstm_kwargs: Dict = None,
                  hidden_units: int = 64,
                  dropout: float = 0.1,
                  label_field: str = 'label'):
    inputs = utils.create_inputs(features)
    input_premise = utils.get_text_inputs(inputs, 'premise')
    input_hypothesis = utils.get_text_inputs(inputs, 'hypothesis')
    embedded_premise = text_embedder(input_premise)
    embedded_hypothesis = text_embedder(input_hypothesis)
    lstm_kwargs = lstm_kwargs if lstm_kwargs else {}
    lstm_kwargs['return_sequences'] = False
    lstm = tf.keras.layers.LSTM(lstm_units, **lstm_kwargs)
    encoded_premise = lstm(embedded_premise)
    encoded_hypothesis = lstm(embedded_hypothesis)
    diff_layer = tf.keras.layers.Lambda(
        lambda x: tf.abs(tf.subtract(x[0], x[1])))
    diff = diff_layer([encoded_premise, encoded_hypothesis])
    if dropout:
        dropout_layer = tf.keras.layers.Dropout(dropout)
        diff = dropout_layer(diff)
    if hidden_units:
        diff = tf.keras.layers.Dense(hidden_units, activation='tanh')(diff)
    probs = tf.keras.layers.Dense(len(targets[label_field].vocab),
                                  activation='softmax',
                                  name=label_field)(diff)
    return tf.keras.models.Model(inputs=list(inputs.values()),
                                 outputs=probs,
                                 name='BiLstmMatcher')
