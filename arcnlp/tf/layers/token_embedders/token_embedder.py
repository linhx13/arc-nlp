# -*- coding: utf-8 -*-


class TokenEmbedder(object):
    def __call__(self, inputs, **kwargs):
        self.call(inputs, **kwargs)

    def call(self, inputs, **kwargs):
        raise NotImplementedError
