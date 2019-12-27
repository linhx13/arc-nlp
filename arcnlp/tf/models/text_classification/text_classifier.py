# -*- coding: utf-8 -*-

from typing import Dict

import tensorflow as tf
import numpy as np

from ..base_model import BaseModel
from arcnlp_tf.data import Field
from arcnlp_tf.layers.text_embedders import TextEmbedder


class TextClassifier(BaseModel):

    def __init__(self,
                 features: Dict[str, Field],
                 targets: Dict[str, Field],
                 text_embedder: TextEmbedder = None,
                 seq2vec_encoder=None,
                 seq2seq_encoder=None,
                 dropout: float = None,
                 label_field: str = "label"):
        super(TextClassifier, self).__init__(features, targets)
        self.text_embedder = text_embedder
        self.seq2vec_encoder = seq2vec_encoder
        self.seq2seq_encoder = seq2seq_encoder
        self.dropout = dropout
        self.label_field = label_field

    def call(self, inputs):
        input_tokens = self._get_text_input(inputs, "tokens")
        embedded_tokens = self.text_embedder(input_tokens)
        if self.seq2seq_encoder:
            embedded_tokens = self.seq2seq_encoder(embedded_tokens)
        encoded_tokens = self.seq2vec_encoder(embedded_tokens)
        if self.dropout:
            encoded_tokens = \
                tf.keras.layers.Dropout(self.dropout)(encoded_tokens)
        probs = tf.keras.layers.Dense(
            len(self.targets[self.label_field].vocab), activation='softmax',
            name=self.label_field)(encoded_tokens)
        return probs

    def decode(self, preds: np.ndarray) -> Dict[str, np.ndarray]:
        classes = [self.targets[self.label_field].vocab.itos[idx]
                   for idx in np.argmax(preds, axis=-1)]
        return {self.label_field: classes}
