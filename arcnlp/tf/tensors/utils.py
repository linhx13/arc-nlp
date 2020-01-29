# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import backend as K


def get_lengths_from_mask(mask: tf.Tensor):
    return K.sum(K.cast(mask, dtype='int32'), axis=1)


def get_last_value(inputs: tf.Tensor, mask: tf.Tensor):
    """Get last value.

    Args:
        inputs: (batch_size, timesteps, embedding_dim)
        mask: (batch_size, timesteps)

    Output shape:
        (batch_size, embedding_dim)
    """
    if mask is None:
        lengths = K.int_shape(inputs)[1]
    else:
        lengths = get_lengths_from_mask(mask)
    lengths -= 1
    lengths = tf.maximum(lengths, tf.zeros_like(lengths, dtype=tf.int32))
    batch_size = tf.shape(lengths)[0]
    batch_nums = tf.range(0, limit=batch_size)  # (batch_size,)
    indices = tf.stack((batch_nums, lengths), axis=1)  # (batch_size, 2)
    res = tf.gather_nd(inputs, indices)  # (batch_size, embedding_dim)
    return res


def cosine_similarity(tensor_1, tensor_2, axis=-1):
    return K.sum(
        K.l2_normalize(tensor_1, axis=axis) *
        K.l2_normalize(tensor_2, axis=axis),
        axis=axis)


def cosine_matrix(x1, x2):
    """Cosine similarity matrix.

    Calculate the cosine similarities between each forward (or backward)
    contextual embedding h_i_p and every forward (or backward)
    contextual embeddings of the other sentence
    # Arguments
        x1: (batch_size, x1_timesteps, embedding_size)
        x2: (batch_size, x2_timesteps, embedding_size)
    # Output shape
        (batch_size, x1_timesteps, x2_timesteps)
    """
    # (batch_size, x1_timesteps, 1, embedding_size)
    x1 = K.expand_dims(x1, axis=2)
    # (batch_size, 1, x2_timesteps, embedding_size)
    x2 = K.expand_dims(x2, axis=1)
    # (batch_size, h1_timesteps, h2_timesteps)
    mat = cosine_similarity(x1, x2)
    return mat


def switch(cond, then_tensor, else_tensor):
    """
    Keras' implementation of K.switch currently uses tensorflow's switch function, which only
    accepts scalar value conditions, rather than boolean tensors which are treated in an
    elementwise function.  This doesn't match with Theano's implementation of switch, but using
    tensorflow's where, we can exactly retrieve this functionality.
    """

    cond_shape = cond.get_shape()
    input_shape = then_tensor.get_shape()
    if cond_shape[-1] != input_shape[-1] and cond_shape[-1] == 1:
        # This happens when the last dim in the input is an embedding dimension. Keras usually does not
        # mask the values along that dimension. Theano broadcasts the value passed along this dimension,
        # but TF does not. Using K.dot() since cond can be a tensor.
        cond = K.dot(tf.cast(cond, tf.float32), tf.ones((1, input_shape[-1])))
    return tf.where(tf.cast(cond, dtype=tf.bool), then_tensor, else_tensor)
