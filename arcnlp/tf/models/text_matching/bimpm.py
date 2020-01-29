# -*- coding: utf-8 -*-

from typing import Dict
from copy import deepcopy

import tensorflow as tf

from .. import utils
from ...data import Field
from ...layers.text_embedders import TextEmbedder
from ...layers import BiMPMatching


def BiMPM(features: Dict[str, Field],
          targets: Dict[str, Field],
          text_embedder: TextEmbedder,
          encoder_type: str = "gru",
          encoder_units: int = 100,
          encoder_kwargs: Dict = None,
          aggregator_type: str = "gru",
          aggregator_units: int = 100,
          aggregator_kwargs: Dict = None,
          dropout: float = 0.1,
          label_field: str = 'label'):
    dropout_layer = tf.keras.layers.Dropout(dropout)

    if encoder_type == 'gru':
        encoder_cls = tf.keras.layers.GRU
    elif encoder_type == 'lstm':
        encoder_cls = tf.keras.layers.LSTM
    else:
        raise ValueError("Unknown encoder_type %s" % encoder_type)
    encoder_kwargs = deepcopy(encoder_kwargs) if encoder_kwargs else {}
    encoder_kwargs['return_sequences'] = True
    encoder = tf.keras.layers.Bidirectional(
        encoder_cls(encoder_units, **encoder_kwargs))

    inputs = utils.create_inputs(features)

    input_premise = utils.get_text_inputs(inputs, 'premise')
    embedded_premise = dropout_layer(text_embedder(input_premise))
    encoded_premise = dropout_layer(encoder(embedded_premise))

    input_hypothesis = utils.get_text_inputs(inputs, 'hypothesis')
    embedded_hypothesis = dropout_layer(text_embedder(input_hypothesis))
    encoded_hypothesis = dropout_layer(encoder(embedded_hypothesis))

    forward_matcher = BiMPMatching(is_forward=True)
    forward_matching_premise, forward_matching_hypothesis = \
        forward_matcher([encoded_premise, encoded_hypothesis])

    backward_matcher = BiMPMatching(is_forward=False)
    backward_matching_premise, backward_matching_hypothesis = \
        backward_matcher([encoded_premise, encoded_hypothesis])

    matching_premise = tf.keras.layers.Concatenate()(
        [forward_matching_premise, backward_matching_premise])
    matching_hypothesis = tf.keras.layers.Concatenate()(
        [forward_matching_hypothesis, backward_matching_hypothesis])

    if aggregator_type == 'gru':
        aggregator_cls = tf.keras.layers.GRU
    elif aggregator_type == 'lstm':
        aggregator_cls = tf.keras.layers.LSTM
    else:
        raise ValueError("Unknown aggregator_type: %s" % aggregator_type)
    aggregator_kwargs = deepcopy(aggregator_kwargs) if aggregator_kwargs else {}
    aggregator_kwargs['return_sequences'] = False
    aggregator = tf.keras.layers.Bidirectional(
        aggregator_cls(aggregator_units, **aggregator_kwargs))

    aggregated_premise = dropout_layer(aggregator(matching_premise))
    aggregated_hypothesis = dropout_layer(aggregator(matching_hypothesis))

    aggregated = tf.keras.layers.Concatenate()(
        [aggregated_premise, aggregated_hypothesis])
    preds = tf.keras.layers.Dense(len(targets[label_field].vocab),
                                  activation='softmax',
                                  name=label_field)(aggregated)
    return tf.keras.models.Model(inputs=list(inputs.values()),
                                 outputs=preds,
                                 name='BiMPM')
