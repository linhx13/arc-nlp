# -*- coding: utf-8 -*-

from typing import Dict

import tensorflow as tf

from ... import utils
from ...data import Field
from ...layers.text_embedders import TextEmbedder


def BasicClassifier(features: Dict[str, Field],
                    targets: Dict[str, Field],
                    text_embedder: TextEmbedder,
                    seq2vec_encoder=None,
                    seq2seq_encoder=None,
                    dropout: float = 0.1,
                    activation='softmax',
                    label_field: str = 'label'):
    inputs = utils.create_inputs(features)
    input_tokens = utils.get_text_inputs(inputs, 'tokens')
    embedded_tokens = text_embedder(input_tokens)
    if seq2seq_encoder:
        embedded_tokens = seq2seq_encoder(embedded_tokens)
    encoded_tokens = seq2vec_encoder(embedded_tokens)
    if dropout:
        encoded_tokens = tf.keras.layers.Dropout(dropout)(encoded_tokens)
    probs = tf.keras.layers.Dense(
        len(targets[label_field].vocab), activation=activation,
        name=label_field)(encoded_tokens)
    return tf.keras.models.Model(inputs=list(inputs.values()),
                                 outputs=probs,
                                 name='BasicClassifier')
