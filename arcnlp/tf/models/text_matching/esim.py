# -*- coding: utf-8 -*-

from typing import Dict
from copy import deepcopy

import tensorflow as tf

from .. import utils
from ...data import Field
from ...layers.text_embedders import TextEmbedder
from ...layers import Attention, BOWEncoder


def ESIM(features: Dict[str, Field],
         targets: Dict[str, Field],
         text_embedder: TextEmbedder,
         lstm_units: int = 128,
         lstm_kwargs: Dict = None,
         hidden_units: int = 64,
         dropout: float = 0.5,
         label_field: str = 'label'):
    """ Implementation of
    `"Enhanced LSTM for Natural Language Inference"
    <https://www.semanticscholar.org/paper/Enhanced-LSTM-for-Natural-Language-Inference-Chen-Zhu/83e7654d545fbbaaf2328df365a781fb67b841b4>`
    by Chen et al., 2017.
    """

    inputs = utils.create_inputs(features)
    input_premise = utils.get_text_inputs(inputs, 'premise')
    input_hypothesis = utils.get_text_inputs(inputs, 'hypothesis')
    embedded_premise = text_embedder(input_premise)
    embedded_hypothesis = text_embedder(input_hypothesis)

    lstm_kwargs = deepcopy(lstm_kwargs) if lstm_kwargs else {}
    lstm_kwargs.pop('return_sequences', None)
    lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_units, return_sequences=True, **lstm_kwargs))
    encoded_premise = lstm(embedded_premise)
    encoded_hypothesis = lstm(embedded_hypothesis)

    aligned_premise = Attention()([encoded_premise, encoded_hypothesis])
    aligned_hypothesis = Attention()([encoded_hypothesis, encoded_premise])

    diff_premise = tf.keras.layers.Subtract()(
        [encoded_premise, aligned_premise])
    mul_premise = tf.keras.layers.Multiply()(
        [encoded_premise, aligned_premise])
    combined_premise = tf.keras.layers.Concatenate()(
        [encoded_premise, aligned_premise, diff_premise, mul_premise])

    diff_hypothesis = tf.keras.layers.Subtract()(
        [encoded_hypothesis, aligned_hypothesis])
    mul_hypothesis = tf.keras.layers.Multiply()(
        [encoded_hypothesis, aligned_hypothesis])
    combined_hypothesis = tf.keras.layers.Concatenate()(
        [encoded_hypothesis, aligned_hypothesis, diff_hypothesis, mul_hypothesis])

    compose_lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_units, return_sequences=True, **lstm_kwargs))
    composed_premise = compose_lstm(combined_premise)
    composed_hypothesis = compose_lstm(combined_hypothesis)

    merged = tf.keras.layers.Concatenate()(
        [BOWEncoder(averaged=True)(composed_premise),
         tf.keras.layers.GlobalMaxPooling1D()(composed_premise),
         BOWEncoder(averaged=True)(composed_hypothesis),
         tf.keras.layers.GlobalMaxPooling1D()(composed_hypothesis)])
    if dropout:
        merged = tf.keras.layers.Dropout(dropout)(merged)
    if hidden_units:
        merged = tf.keras.layers.Dense(hidden_units, activation='tanh')(merged)
    probs = tf.keras.layers.Dense(len(targets[label_field].vocab),
                                  activation='softmax',
                                  name=label_field)(merged)
    return tf.keras.models.Model(inputs=list(inputs.values()),
                                 outputs=probs,
                                 name="ESIM")
