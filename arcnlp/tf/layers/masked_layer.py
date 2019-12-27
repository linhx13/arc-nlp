# -*- coding: utf-8 -*-

import tensorflow as tf


class MaskedLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaskedLayer, self).__init__(**kwargs)

    def call(self, inputs, mask=None):  # pylint: disable=arguments-differ
        raise NotImplementedError
