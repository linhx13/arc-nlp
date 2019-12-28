# -*- coding: utf-8 -*-

from typing import Dict, Iterable

import tensorflow as tf

from .. import utils
from ...data import Field
from ...layers.text_embedders import TextEmbedder
from ...layers.seq2vec_encoders import CNNEncoder


def TextCNN(features: Dict[str, Field],
            targets: Dict[str, Field],
            text_embedder: TextEmbedder = None,
            filters: int = 100,
            kernel_sizes: Iterable[int] = (2, 3, 4, 5),
            conv_layer_activation='relu',
            l1_regularization: float = None,
            l2_regularization: float = None,
            dropout: float = 0.1,
            label_field: str = 'label'):
    inputs = utils.create_inputs(features)
    input_tokens = utils.get_text_inputs(inputs, 'tokens')
    embedded_tokens = text_embedder(input_tokens)
    cnn_encoder = CNNEncoder(filters, kernel_sizes,
                             conv_layer_activation,
                             l1_regularization,
                             l2_regularization)
    encoded_tokens = cnn_encoder(embedded_tokens)
    if dropout:
        encoded_tokens = tf.keras.layers.Dropout(dropout)(encoded_tokens)
    probs = tf.keras.layers.Dense(
        len(targets[label_field].vocab), activation='softmax',
        name=label_field)(encoded_tokens)
    return tf.keras.models.Model(inputs=list(inputs.values()),
                                 outputs=probs,
                                 name='TextCNN')
