# -*- coding: utf-8 -*-

from typing import Dict

import tensorflow as tf
from tensorflow.keras import backend as K

from ... import utils
from ...data import Field
from ...layers.text_embedders import TextEmbedder


def RCNNClassifier(features: Dict[str, Field],
                   targets: Dict[str, Field],
                   text_embedder: TextEmbedder,
                   rnn_type: str = 'lstm',
                   rnn_units: int = 128,
                   rnn_kwargs: Dict = None,
                   hidden_units: int = 100,
                   dropout: float = 0.1,
                   activation='softmax',
                   label_field: str = 'label'):
    inputs = utils.create_inputs(features)
    input_tokens = utils.get_text_inputs(inputs, 'tokens')
    embedded_tokens = text_embedder(input_tokens)
    rnn_kwargs = rnn_kwargs if rnn_kwargs is not None else {}
    rnn_class = get_rnn_class(rnn_type)
    forward_layer = rnn_class(rnn_units, return_sequences=True,
                              name='forward_layer', **rnn_kwargs)
    backword_layer = rnn_class(rnn_units, return_sequences=True,
                               go_backwards=True, name='backword_layer', **rnn_kwargs)
    forward_seqs = forward_layer(embedded_tokens)
    left_context = tf.keras.layers.Lambda(
        shift_right, mask=embedded_tokens._keras_mask, name='left_context')(forward_seqs)
    backword_seqs = backword_layer(embedded_tokens)
    backword_seqs = tf.keras.layers.Lambda(
        lambda x: tf.keras.backend.reverse(x, 1), mask=embedded_tokens._keras_mask)(backword_seqs)
    right_context = tf.keras.layers.Lambda(
        shift_left, mask=embedded_tokens._keras_mask, name='right_context')(backword_seqs)
    embedded_tokens = tf.keras.layers.Concatenate()(
        [left_context, embedded_tokens, right_context])
    if hidden_units:
        embedded_tokens = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(hidden_units, activation='tanh'))(embedded_tokens)
    encoded_tokens = tf.keras.layers.GlobalMaxPooling1D()(embedded_tokens)
    if dropout:
        encoded_tokens = tf.keras.layers.Dropout(dropout)(encoded_tokens)
    probs = tf.keras.layers.Dense(
        len(targets[label_field].vocab), activation=activation,
        name=label_field)(encoded_tokens)
    return tf.keras.models.Model(inputs=list(inputs.values()),
                                 outputs=probs,
                                 name="RCNNClassifier")


def shift_left(x, offset=1):
    assert offset > 0
    return K.concatenate([x[:, offset:], K.zeros_like(x[:, :offset])], axis=1)


def shift_right(x, offset=1):
    assert offset > 0
    return K.concatenate([K.zeros_like(x[:, :offset]), x[:, :-offset]], axis=1)


def get_rnn_class(rnn_type):
    if rnn_type == 'rnn':
        return tf.keras.layers.RNN
    elif rnn_type == 'lstm':
        return tf.keras.layers.LSTM
    elif rnn_type == 'gru':
        return tf.keras.layers.GRU
    else:
        raise ValueError("Invalid rnn type: %s" % rnn_type)
