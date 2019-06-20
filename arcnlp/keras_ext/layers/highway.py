# -*- coding: utf-8 -*-

from keras.layers import Highway as KerasHighway


class Highway(KerasHighway):
    def __ini__(self, **kwargs):
        super(Highway, self).__init__(**kwargs)
        self.supports_masking = True
