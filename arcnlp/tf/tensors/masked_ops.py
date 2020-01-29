# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import backend as K

from .utils import switch


def masked_max(x: tf.Tensor, mask: tf.Tensor, axis: int = -1,
               keepdims: bool = False):
    return K.max(x, axis=axis, keepdims=keepdims)


def masked_mean(input: tf.Tensor, mask: tf.Tensor, axis: int = -1,
                keepdims: bool = False):
    if mask is None:
        return K.mean(input, axis=axis, keepdims=keepdims)
    mask = K.cast(mask, K.floatx())
    if K.ndim(mask) < K.ndim(input):
        mask = K.expand_dims(mask)
    input *= mask
    output = K.sum(input, axis=axis, keepdims=keepdims) / \
        (K.sum(mask, axis=axis, keepdims=keepdims) + K.epsilon())
    return output


def masked_softmax(vector, mask):
    """
    `K.softmax(vector)` does not work if some elements of `vector` should be masked.  This performs
    a softmax on just the non-masked portions of `vector` (passing None in for the mask is also
    acceptable; you'll just get a regular softmax).

    We assume that both `vector` and `mask` (if given) have shape (batch_size, vector_dim).

    In the case that the input vector is completely masked, this function returns an array
    of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of a model
    that uses categorial cross-entropy loss.
    """
    # We calculate masked softmax in a numerically stable fashion, as done
    # in https://github.com/rkadlec/asreader/blob/master/asreader/custombricks/softmax_mask_bricks.py
    if mask is not None:
        # Here we get normalized log probabilities for
        # enhanced numerical stability.
        mask = K.cast(mask, "float32")
        input_masked = mask * vector
        shifted = mask * (input_masked - K.max(input_masked, axis=1,
                                               keepdims=True))
        # We add epsilon to avoid numerical instability when
        # the sum in the log yields 0.
        normalization_constant = K.log(K.sum(mask * K.exp(shifted), axis=1,
                                             keepdims=True) + K.epsilon())
        normalized_log_probabilities = mask * \
            (shifted - normalization_constant)
        unmasked_probabilities = K.exp(normalized_log_probabilities)
        return switch(mask, unmasked_probabilities, K.zeros_like(unmasked_probabilities))
    else:
        # There is no mask, so we use the provided ``K.softmax`` function.
        return K.softmax(vector)
