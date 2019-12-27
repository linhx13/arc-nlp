# -*- coding: utf-8 -*-

from typing import Dict

import tensorflow as tf
import numpy as np

from .base_model import BaseModel
from arcnlp_tf.data import Field
from arcnlp_tf.layers.text_embedders import BasicTextEmbedder


class SimpleTagger(BaseModel):

    def __init__(self,
                 features: Dict[str, Field],
                 targets: Dict[str, Field],
                 token_embedders: Dict,
                 feature_embedders: Dict,
                 encoder,
                 dropout: float = None,
                 **kwargs):
        super(SimpleTagger, self).__init__(features, targets)
        # Subtract 2 to ignore the unk and pad tokens in tag_field vocab
        self.num_tags = len(self.fields['tags'].vocab) - 2
        self.text_embedder = BasicTextEmbedder(token_embedders)
        self.feature_embedders = feature_embedders
        self.encoder = encoder
        self.dropout = dropout

    def call(self, inputs):
        embedded = []
        input_tokens = self._get_text_input(inputs, "tokens")
        embedded_tokens = self.text_embedder(input_tokens)
        embedded.append(embedded_tokens)

        if self.feature_embedders:
            for name, embedder in self.feature_embedders.items():
                input_fea = self._get_input(inputs, name)
                embedded.append(embedder(input_fea))
        embedded = tf.keras.layers.Concatenate()(embedded)
        if self.dropout:
            embedded = tf.keras.layers.Dropout(self.dropout)(embedded)

        encoded = self.encoder(embedded)
        if self.dropout:
            encoded = tf.keras.layers.Dropout(self.dropout)(encoded)
        projection_layer = tf.keras.layers.Dense(self.num_tags,
                                                 activation='softmax')
        predicted_tags = tf.keras.layers.TimeDistributed(
            projection_layer, name='tags')(encoded)
        return predicted_tags

    def decode(self, preds: np.ndarray) -> Dict[str, np.ndarray]:
        tags = [[self.targets['tags'].vocab.itos[idx + 2] for idx in tag_idx]
                for tag_idx in np.argmax(preds, axis=-1)]
        return {'tags': tags}
