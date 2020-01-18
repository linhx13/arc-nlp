# -*- coding: utf-8 -*-

from typing import Dict

import tensorflow as tf

from .. import utils
from ...data import Field
from ...layers.text_embedders import TextEmbedder


def SimpleTagger(features: Dict[str, Field],
                 targets: Dict[str, Field],
                 text_embedder: TextEmbedder,
                 encoder,
                 feature_embedders: Dict = None,
                 dropout: float = 0.1,
                 **kwargs):
    inputs = utils.create_inputs(features)
    embedded = []
    input_tokens = utils.get_text_inputs(inputs, 'tokens')
    embedded_tokens = text_embedder(input_tokens)
    embedded.append(embedded_tokens)

    if feature_embedders:
        for name, embedder in feature_embedders.items():
            embedded.append(embedder(inputs[name]))
    if len(embedded) > 1:
        embedded = tf.keras.layers.Concatenate()(embedded)
    else:
        embedded = embedded[0]
    if dropout:
        embedded = tf.keras.layers.Dropout(dropout)(embedded)
    encoded = encoder(embedded)
    if dropout:
        encoded = tf.keras.layers.Dropout(dropout)(encoded)

    # Subtract 2 to ignore the unk and pad tokens in tag_field vocab
    num_tags = len(targets['tags'].vocab) - 2
    proj_layer = tf.keras.layers.Dense(num_tags, activation='softmax')
    preds = tf.keras.layers.TimeDistributed(proj_layer, name='tags')(encoded)
    return tf.keras.models.Model(inputs=list(inputs.values()),
                                 outputs=preds,
                                 name='SimpleTagger')
