# -*- coding: utf-8 -*-


class TextEmbedder(object):
    def __call__(self, inputs, **kwargs):
        return self.call(inputs, **kwargs)

    def call(self, inputs, **kwargs):
        raise NotImplementedError
