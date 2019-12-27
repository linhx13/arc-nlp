# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import backend as K


def amsoftmax_loss(y_true, y_pred, scale=30, margin=0.35):
    """ Reference: https://kexue.fm/archives/5743
    Note: y_pred should be a cosin value.
    """

    y_pred = y_true * (y_pred - margin) + (1 - y_true) * y_pred
    y_pred *= scale
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)


def sparse_amsoftmax_loss(y_true, y_pred, scale=30, margin=0.35):
    y_true = K.expand_dims(y_true[:, 0], 1)
    y_true = K.cast(y_true, 'int32')
    batch_idxs = K.arange(0, K.shape(y_true)[0])
    batch_idxs = K.expand_dims(batch_idxs, 1)
    idxs = K.concatenate([batch_idxs, y_true], 1)
    y_true_pred = tf.gather_nd(y_pred, idxs)
    y_true_pred = K.expand_dims(y_true_pred, 1)
    y_true_pred_margin = y_true_pred - margin
    _Z = K.concatenate([y_pred, y_true_pred_margin], 1)
    _Z = _Z * scale
    logZ = K.logsumexp(_Z, 1, keepdims=True)
    logZ = logZ + K.log(1 - K.exp(scale * y_true_pred - logZ))
    return - y_true_pred_margin * scale + logZ
