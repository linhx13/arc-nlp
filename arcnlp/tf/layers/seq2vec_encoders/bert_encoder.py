# -*- coding: utf-8 -*-

import tensorflow as tf
import os


os.environ['TF_KERAS'] = '1'


class BertEncoder(tf.keras.layers.Layer):

    def __init__(self, model_path: str, trainable: bool = True,
                 seq_len: int = None, **kwargs):
        super(BertEncoder, self).__init__(**kwargs)
        self.supports_masking = True
        self.model_path = model_path
        self.trainable = trainable
        self.seq_len = seq_len

    def build(self, input_shape):
        import keras_bert
        paths = keras_bert.get_checkpoint_paths(self.model_path)
        self.bert_model = keras_bert.load_trained_model_from_checkpoint(
            paths.config, paths.checkpoint, trainable=self.trainable,
            seq_len=self.seq_len)

    def get_config(self):
        config = {"model_path": self.model_path,
                  "trainable": self.trainable,
                  "seq_len": self.seq_len}
        base_config = super(BertEncoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        bert_output_shape = self.model.output_shape
        return (input_shape[0], bert_output_shape[-1])

    def call(self, inputs, mask=None):
        return self.bert_model(inputs)[:, 0]
