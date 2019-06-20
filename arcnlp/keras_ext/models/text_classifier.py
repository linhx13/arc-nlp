# -*- coding: utf-8 -*-

from tensorflow import keras

from .model import Model


class TextClassifier(Model):

    def __init__(self, text_field_embedder, seq2vec_encoder,
                 seq2seq_encoder=None, dropout=None, num_labels=None):
        super(TextClassifier, self).__init__()
        self.text_field_embedder = text_field_embedder
        self.seq2vec_encoder = seq2vec_encoder
        self.seq2seq_encoder = seq2seq_encoder
        self.dropout = dropout
        self.num_labels = num_labels

        self.text_field = {}  # str -> Field
        self.label_field = None  # Field

    def build_model(self):
        input_text = {k: keras.layers.Input(shape=(f.fix_length,))
                      for k, f in self.text_field.items()}
        input_text = sorted(input_text.items(), key=lambda t: t[0])
        embedded_text = self.text_field_embedder(input_text)
        if self.seq2seq_encoder:
            embedded_text = self.seq2seq_encoder(embedded_text)
        embedded_text = self.seq2vec_encoder(embedded_text)
        if self.dropout:
            embedded_text = keras.layers.Dropout(self.dropout)(embedded_text)
        logits = keras.layers.Dense(self.num_labels)(embedded_text)
        probs = keras.layers.Activation("softmax")(logits)
        return keras.models.Model(input_text, outputs=probs)
