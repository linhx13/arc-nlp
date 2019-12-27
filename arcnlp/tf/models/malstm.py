# -*- coding: utf-8 -*-

from typing import Dict

import tensorflow as tf
from tensorflow.keras import backend as K

from .base_model import BaseModel
from arcnlp_tf.data import Field
from arcnlp_tf.layers.text_embedders import TextEmbedder


class ExpNegManhattanDistance(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(ExpNegManhattanDistance, self).__init__(**kwargs)
        self.supports_masking = False

    def call(self, inputs):
        left, right = inputs
        return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)


def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))


class MaLSTM(BaseModel):
    """ Implementation of
    Siamese Recurrent Architectures for Learning Sentence Similarity
    """

    def __init__(self,
                 features: Dict[str, Field],
                 targets: Dict[str, Field],
                 text_embedder: TextEmbedder = None,
                 lstm_units: int = 50,
                 dropout: float = None,
                 **kwargs):
        super(MaLSTM, self).__init__(features, targets)

        self.text_embedder = text_embedder
        self.lstm_units = lstm_units
        self.dropout = dropout

    def call(self, inputs):
        input_premise = self._get_text_input(inputs, "premise")
        input_hypothesis = self._get_text_input(inputs, "hypothesis")

        embedded_premise = self.text_embedder(input_premise)
        embedded_hypothesis = self.text_embedder(input_hypothesis)
        if self.dropout:
            embedded_premise = tf.keras.layers.Dropout(
                self.dropout)(embedded_premise)
            embedded_hypothesis = tf.keras.layers.Dropout(
                self.dropout)(embedded_hypothesis)

        shared_lstm = tf.keras.layers.LSTM(self.lstm_units)
        encoded_premise = shared_lstm(embedded_premise)
        encoded_hypothesis = shared_lstm(embedded_hypothesis)

        malstm_distance = ExpNegManhattanDistance(name='label')(
            [encoded_premise, encoded_hypothesis])
        return malstm_distance

    def get_custom_objects(self):
        custom_objects = super(MaLSTM, self).get_custom_objects()
        custom_objects[ExpNegManhattanDistance.__name__] = ExpNegManhattanDistance
        return custom_objects
