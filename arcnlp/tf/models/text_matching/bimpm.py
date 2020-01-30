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
          with_full_match: bool = True,
          with_maxpool_match: bool = True,
          with_attentive_match: bool = True,
          with_max_attentive_match: bool = True,
          aggregator_type: str = "gru",
          aggregator_units: int = 100,
          aggregator_kwargs: Dict = None,
          dropout: float = 0.1,
          label_field: str = 'label'):
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
    embedded_premise = text_embedder(input_premise)
    if dropout:
        embedded_premise = tf.keras.layers.Dropout(dropout)(embedded_premise)
    encoded_premise = encoder(embedded_premise)
    if dropout:
        encoded_premise = tf.keras.layers.Dropout(dropout)(encoded_premise)

    input_hypothesis = utils.get_text_inputs(inputs, 'hypothesis')
    embedded_hypothesis = text_embedder(input_hypothesis)
    if dropout:
        embedded_hypothesis = tf.keras.layers.Dropout(
            dropout)(embedded_hypothesis)
    encoded_hypothesis = encoder(embedded_hypothesis)
    if dropout:
        encoded_hypothesis = tf.keras.layers.Dropout(
            dropout)(encoded_hypothesis)

    matcher_kwargs = {
        'with_full_match': with_full_match,
        'with_maxpool_match': with_maxpool_match,
        'with_attentive_match': with_attentive_match,
        'with_max_attentive_match': with_max_attentive_match
    }

    forward_matcher = BiMPMatching(is_forward=True, **matcher_kwargs)
    forward_matching_premise, forward_matching_hypothesis = \
        forward_matcher([encoded_premise, encoded_hypothesis])

    backward_matcher = BiMPMatching(is_forward=False, **matcher_kwargs)
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
    aggregator_kwargs = deepcopy(
        aggregator_kwargs) if aggregator_kwargs else {}
    aggregator_kwargs['return_sequences'] = False
    aggregator = tf.keras.layers.Bidirectional(
        aggregator_cls(aggregator_units, **aggregator_kwargs))

    aggregated_premise = aggregator(matching_premise)
    if dropout:
        aggregated_premise = tf.keras.layers.Dropout(
            dropout)(aggregated_premise)
    aggregated_hypothesis = aggregator(matching_hypothesis)
    if dropout:
        aggregated_hypothesis = tf.keras.layers.Dropout(
            dropout)(aggregated_hypothesis)

    aggregated = tf.keras.layers.Concatenate()(
        [aggregated_premise, aggregated_hypothesis])
    preds = tf.keras.layers.Dense(len(targets[label_field].vocab),
                                  activation='softmax',
                                  name=label_field)(aggregated)
    return tf.keras.models.Model(inputs=list(inputs.values()),
                                 outputs=preds,
                                 name='BiMPM')
