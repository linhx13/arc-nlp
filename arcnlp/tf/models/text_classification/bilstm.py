# -*- coding: utf-8 -*-

from typing import Dict

import tensorflow as tf

from ... import utils
from ...data import Field
from ...layers.text_embedders import TextEmbedder


def BiLstmClassifier(features: Dict[str, Field],
                     targets: Dict[str, Field],
                     text_embedder: TextEmbedder,
                     lstm_units: int = 128,
                     lstm_kwargs: Dict = None,
                     hidden_units: int = 100,
                     dropout: float = 0.1,
                     activation='softmax',
                     label_field: str = 'label'):
    inputs = utils.create_inputs(features)
    input_tokens = utils.get_text_inputs(inputs, 'tokens')
    embedded_tokens = text_embedder(input_tokens)
    lstm_kwargs = lstm_kwargs if lstm_kwargs is not None else {}
    lstm = tf.keras.layers.LSTM(lstm_units, **lstm_kwargs)
    encoded_tokens = tf.keras.layers.Bidirectional(lstm)(embedded_tokens)
    if dropout:
        encoded_tokens = tf.keras.layers.Dropout(dropout)(encoded_tokens)
    if hidden_units:
        encoded_tokens = tf.keras.layers.Dense(
            hidden_units, activation='tanh')(encoded_tokens)
    probs = tf.keras.layers.Dense(
        len(targets[label_field].vocab), activation=activation,
        name=label_field)(encoded_tokens)
    return tf.keras.models.Model(inputs=list(inputs.values()),
                                 outputs=probs,
                                 name="BiLstmClassifier")
