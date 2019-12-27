# -*- coding: utf-8 -*-

from typing import Dict

import tensorflow as tf
import numpy as np

from .base_model import BaseModel
from arcnlp_tf.data import Field
from arcnlp_tf.layers.text_embedders import TextEmbedder
from arcnlp_tf.layers import CRF
from arcnlp_tf.losses import crf_loss
from arcnlp_tf.metrics import crf_accuracy


class CrfTagger(BaseModel):

    def __init__(self,
                 features: Dict[str, Field],
                 targets: Dict[str, Field],
                 tokens_embedder: TextEmbedder = None,
                 feature_embedders: Dict = None,
                 encoder=None,
                 dropout: float = None,
                 sparse_target=True,
                 **kwargs):
        super(CrfTagger, self).__init__(features, targets)
        # Subtract 2 to ignore the unk and pad tokens in tag_field vocab
        self.num_tags = len(self.targets['tags'].vocab) - 2
        self.tokens_embedder = tokens_embedder
        self.feature_embedders = feature_embedders
        self.encoder = encoder
        self.dropout = dropout
        self.sparse_target = sparse_target

    def call(self, inputs):
        embedded = []
        input_tokens = self._get_text_input(inputs, "tokens")
        embedded_tokens = self.tokens_embedder(input_tokens)
        embedded.append(embedded_tokens)

        if self.feature_embedders:
            for name, embedder in self.feature_embedders.items():
                input_fea = self._get_input(inputs, name)
                embedded.append(embedder(input_fea))
        if len(embedded) > 1:
            embedded = tf.keras.layers.Concatenate()(embedded)
        else:
            embedded = embedded[0]
        if self.dropout:
            embedded = tf.keras.layers.Dropout(self.dropout)(embedded)

        encoded = self.encoder(embedded)
        if self.dropout:
            encoded = tf.keras.layers.Dropout(self.dropout)(encoded)
        crf = CRF(self.num_tags, name='tags', sparse_target=self.sparse_target)
        predicted_tags = crf(encoded)
        return predicted_tags

    def decode(self, preds: np.ndarray) -> Dict[str, np.ndarray]:
        tags = [[self.targets['tags'].vocab.itos[idx + 2] for idx in tag_idx]
                for tag_idx in np.argmax(preds, axis=-1)]
        return {'tags': tags}

    @classmethod
    def get_custom_objects(cls):
        custom_objects = super(CrfTagger, cls).get_custom_objects()
        custom_objects['CRF'] = CRF
        custom_objects['crf_loss'] = crf_loss
        custom_objects['crf_accuracy'] = crf_accuracy
        return custom_objects
