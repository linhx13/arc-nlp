# -*- coding: utf-8 -*-

from tensorflow import keras

from .model import Model


class TextClassifier(Model):

    def __init__(self, text_fields, label_field, text_field_embedder,
                 seq2vec_encoder, seq2seq_encoder=None, dropout=None):
        super(TextClassifier, self).__init__()
        self.text_fields = text_fields
        self.label_field = label_field
        self.text_field_embedder = text_field_embedder
        self.seq2vec_encoder = seq2vec_encoder
        self.seq2seq_encoder = seq2seq_encoder
        self.dropout = dropout

    def build_model(self):
        input_text = {k: keras.layers.Input(shape=(f.fix_length,), name=k)
                      for k, f in self.text_fields.items()}
        print('input_text:', input_text)
        embedded_text = self.text_field_embedder(input_text)
        if self.seq2seq_encoder:
            embedded_text = self.seq2seq_encoder(embedded_text)
        embedded_text = self.seq2vec_encoder(embedded_text)
        if self.dropout:
            embedded_text = keras.layers.Dropout(self.dropout)(embedded_text)
        logits = keras.layers.Dense(len(self.label_field.vocab))(embedded_text)
        probs = keras.layers.Activation("softmax")(logits)

        inputs = sorted([(k, v) for k, v in input_text.items()],
                        key=lambda x: x[0])
        inputs = [x[1] for x in inputs]
        return keras.models.Model(inputs, outputs=probs)
