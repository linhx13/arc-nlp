# -*- coding: utf-8 -*-

from typing import Dict, Any
from deepcopy import deepcopy

import tensorflow as tf
from tensorflow.keras import backend as K

from ...tensors.similarity_functions import similarity_functions
from ...tensors.masked_operations import masked_softmax


class Attention(tf.keras.layers.Layer):
    """This Layer takes two inputs: a (batched) vector and a matrix. We compute
    the similarity  between the vector and each row in the matrix, and then
    (optionally) perform a softmax over rows using those computed similarities.
    We handle masking properly for masked rows in the matrix, though we ignore
    any masking on the vector.

    By default similarity is computed with a dot product, but you can
    alternatively use a parameterized similarity function if you wish.

    Input:

    - vector: shape ``(batch_size, embedding_dim)``, mask is ignored if provided
    - matrix: shape ``(batch_size, num_rows, embedding_dim)``, with mask
      ``(batch_size, num_rows) ``

    Output:

    - attention: shape ``(batch_size, num_rows)``. If ``normalize`` is ``True``,
      we returned no mask, as we've already applied it (masked input rows have
      vluae 0 in the output). If ``normalize`` is ``False``, we return the
      matrix mask, if there was one.

    Parameters
    ----------
    similarity_function_params: ``Dict[str, Any]``, optional (default: ``{}``)
        These parameters get passed to a similarity function. The default
        similarity function with no parameters is a simple dot product.
    normalize: ``bool``, optional (default: ``True``)
        If true, we normalize the computed similarities with a softmax, to
        return a probability distribution for your attention. If false, this is
        just computing a similarity score.
    """

    def __init__(self, similarity_function_params: Dict[str, Any] = None,
                 normalize: bool = True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.similarity_function_params = deepcopy(similarity_function_params)
        sim_function_params = deepcopy(similarity_function_params)
        sim_function_choice = sim_function_params.pop('type', 'dot_product')
        sim_function_params['name'] = self.name + '_similarity_function'
        self.similarity_function = \
            similarity_functions[sim_function_choice](**sim_function_params)
        self.normalize = normalize

    def build(self, input_shape):
        tensor_1_dim = input_shape[0][-1]
        tensor_2_dim = input_shape[1][-1]
        self.trainable_weights = self.sim_func.initialize_weights(tensor_1_dim, tensor_2_dim)
        super(Attention, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        if self.normalize or mask is None:
            return None
        return mask[1]

    def compute_output_shape(self, input_shape):
        return (input_shape[1][0], input_shape[1][1])

    def call(self, inputs, mask=None):
        vector, matrix = inputs
        if mask is None:
            matrix_mask = None
        else:
            matrix_mask = mask[1]
        num_rows = K.int_shape(matrix)[1]
        tiled_vector = K.repeat_elements(K.expand_dims(vector, axis=1),
                                         num_rows, axis=1)
        sims = self.similarity_function.compute_similarity(tiled_vector,
                                                           matrix)
        if self.normalize:
            return masked_softmax(sims, matrix_mask)
        else:
            return sims

    def get_config(self):
        base_config = super(Attention, self).get_config()
        config = {
            "similarity_function_params": self.similarity_function_params,
            "normalize": self.normalize
        }
        base_config.update(config)
        return config
