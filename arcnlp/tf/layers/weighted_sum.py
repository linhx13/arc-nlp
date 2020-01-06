# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import backend as K


class WeightedSum(tf.keras.layers.Layer):
    """ This `Layer` takes a matrix of vectors and a vector of row weights, and
    returns a weighted sum of the vectors.

    Inputs:

    - matrix: (batch_size, num_rows, embedding_dim), with mask (batch_size, num_rows)
    - vector: (batch_size, num_rows), mask is ignored

    Outputs:

    - A weighted sum of the rows in the matrix, with shape (batch_size, embedding_dim),
      with mask is None.
    """

    def __init__(self, use_masking: bool = True, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.use_masking = use_masking

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shapes):
        matrix_shape, attention_shape = input_shapes
        return attention_shape[:-1] + matrix_shape[-1:]

    def call(self, inputs, mask=None):
        matrix, attention_vector = inputs
        num_attention_dims = K.ndim(attention_vector)
        num_matrix_dims = K.ndim(matrix) - 1
        for _ in range(num_attention_dims - num_matrix_dims):
            matrix = K.expand_dims(matrix, axis=1)
        if mask is None:
            matrix_mask = None
        else:
            matrix_mask = mask[0]
        if self.use_masking and matrix_mask is not None:
            for _ in range(num_attention_dims - num_matrix_dims):
                matrix_mask = K.expand_dims(matrix_mask, axis=1)
            matrix = K.cast(K.expand_dims(matrix_mask), 'float32') * matrix
        return K.sum(K.expand_dims(attention_vector, axis=-1) * matrix, -2)
