# -*- coding: utf-8 -*-

from keras import backend as K
from keras.engine import InputSpec

from ..masked_layer import MaskedLayer


class BOWEncoder(MaskedLayer):
    """Bag of words encoder takes a matrix of shape (num_words, word_dim) and
    returns a vector of size (word_dim), which simply sums or averages the
    (unmasked) rows in the input matrix.
    """

    def __init__(self, averaged=False, **kwargs):
        self.input_spec = [InputSpec(ndim=3)]
        self.averaged = averaged
        kwargs.pop("units", None)
        super(BOWEncoder, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def compute_mask(self, inputs, mask=None):
        # We need to override this method because layer passes the input mask
        # unchanged since the layer supports masking. We don't want that. After
        # the input is summed (or averaged), we can stop propagating the mask.
        return None

    def call(self, inputs, mask=None):
        if mask is None:
            if self.averaged:
                return K.mean(inputs, axis=1)
            else:
                return K.sum(inputs, axis=1)
        else:
            # Mask (samples, num_words) has 0s for masked elements and 1s
            # everywhere else.
            mask = K.cast(mask, K.floatx())
            if K.ndim(mask) < K.ndim(inputs):
                mask = K.expand_dims(mask)
            inputs *= mask
            output = K.sum(inputs, axis=1)
            if self.averaged:
                output /= (K.sum(mask, axis=1) + K.epsilon())
            return output

    def get_config(self):
        config = {"averaged": self.averaged}
        base_config = super(BOWEncoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
