# -*- coding: utf-8 -*-

from typing import Dict, Any
from copy import deepcopy

import tensorflow as tf
from tensorflow.keras import backend as K

from arcnlp_tf.tensors.similarity_functions import similarity_functions


class MatrixAttention(tf.keras.layers.Layer):
    """
    This ``Layer`` takes two matrics as input and returns a matrix of attentions.

    Input:
        - matrix_1: ``(batch_size, num_rows_1, embedding_dim)``, with mask
            ``(batch_size, num_rows_1)``
        - matrix_2: ``(batch_size, num_rows_2, embedding_dim)``, with masked
            ``(batch_size, num_rows_2)``

    Output:
        - ``(batch_size, num_rows_1, num_rows_2)``, with mask of same shape

    Args:
        sim_func_params: Dict[str, Any], default={}
    """

    def __init__(self, sim_func_params: Dict[str, Any] = None, **kwargs):
        super(MatrixAttention, self).__init__(**kwargs)
        self.sim_func_params = deepcopy(sim_func_params)
        sim_func_params = deepcopy(sim_func_params) if sim_func_params else {}
        sim_func_type = sim_func_params.pop('type', 'dot_product')
        sim_func_params['name'] = self.name + '_similarity_function'
        self.sim_func = similarity_functions[sim_func_type](**sim_func_params)

    def build(self, input_shape):
        tensor_1_dim = input_shape[0][-1]
        tensor_2_dim = input_shape[1][-1]
        self.trainable_weights = self.sim_func.initialize_weights(
            tensor_1_dim, tensor_2_dim)
        super(MatrixAttention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[1][1])

    def compute_mask(self, inputs, mask=None):
        mask_1, mask_2 = mask
        if mask_1 is None and mask_2 is None:
            return None
        if mask_1 is None:
            mask_1 = K.ones_like(K.sum(inputs[0], axis=-1))
        if mask_2 is None:
            mask_2 = K.ones_like(K.sum(inputs[1], axis=-1))
        mask_1 = K.expand_dims(mask_1, axis=2)
        mask_2 = K.expand_dims(mask_2, axis=1)
        return K.cast(K.batch_dot(mask_1, mask_2), 'uint8')

    def call(self, inputs, mask=None):
        matrix_1, matrix_2 = inputs
        num_rows_1 = K.shape(matrix_1)[1]
        num_rows_2 = K.shape(matrix_2)[1]
        tile_dims_1 = K.concatenate([[1, 1], [num_rows_2], [1]], 0)
        tile_dims_2 = K.concatenate([[1], [num_rows_1], [1, 1]], 0)
        tiled_matrix_1 = K.tile(K.expand_dims(matrix_1, axis=2), tile_dims_1)
        tiled_matrix_2 = K.tile(K.expand_dims(matrix_2, axis=1), tile_dims_2)
        return self.sim_func.compute_similarity(tiled_matrix_1, tiled_matrix_2)

    def get_config(self):
        base_config = super(MatrixAttention, self).get_config()
        config = {'sim_func_params': self.sim_func_params}
        base_config.update(config)
        return base_config
