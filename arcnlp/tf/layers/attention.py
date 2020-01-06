# -*- coding: utf-8 -*-

import tensorflow as tf


class Attention(tf.keras.layers.Attention):

    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def get_config(self):
        config = {'use_scale': self.use_scale,
                  'causal': self.causal}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
