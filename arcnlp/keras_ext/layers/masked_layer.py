# -*- coding: utf-8 -*-

from keras.layers import Layer


class MaskedLayer(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaskedLayer, self).__init__(**kwargs)

    def call(self, inputs, mask=None):  # pylint: disable=arguments-differ
        raise NotImplementedError
