# -*- coding: utf-8 -*-

from typing import Dict

import tensorflow as tf

from ... import utils
from ...data import Field
from ...layers.text_embedders import TextEmbedder
from ...layers.seq2vec_encoders import BOWEncoder


def BOWClassifier(
        features: Dict[str, Field],
        targets: Dict[str, Field],
        text_embedder: TextEmbedder,
        hidden_dim: int = 128,
        hidden_dim2: int = 96,
        dropout: float = 0.1,
        activation='softmax',
        label_field: str = 'label'):
    inputs = utils.create_inputs(features)
    input_tokens = utils.get_text_inputs(inputs, 'tokens')
    embedded_tokens = text_embedder(input_tokens)
    encoded_tokens = BOWEncoder()(embedded_tokens)
    if dropout:
        encoded_tokens = tf.keras.layers.Dropout(dropout)(encoded_tokens)
    if hidden_dim:
        encoded_tokens = tf.keras.layers.Dense(
            hidden_dim, activation='tanh')(encoded_tokens)
    if hidden_dim2:
        encoded_tokens = tf.keras.layers.Dense(
            hidden_dim2, activation='tanh')(encoded_tokens)
    probs = tf.keras.layers.Dense(
        len(targets[label_field].vocab), activation=activation,
        name=label_field)(encoded_tokens)
    return tf.keras.models.Model(inputs=list(inputs.values()),
                                 outputs=probs,
                                 name="BOWClassifier")
