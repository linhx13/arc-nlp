# -*- coding: utf-8 -*-

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import backend as K

from ..tensors.utils import (
    get_lengths_from_mask,
    get_last_value,
    cosine_similarity,
    cosine_matrix)
from ..tensors.masked_ops import masked_max, masked_mean, masked_softmax


class BiMPMatching(tf.keras.layers.Layer):

    def __init__(self,
                 num_perspectives: int = 20,
                 share_weights_between_directions: bool = True,
                 is_forward: bool = None,
                 with_full_match: bool = True,
                 with_maxpool_match: bool = True,
                 with_attentive_match: bool = True,
                 with_max_attentive_match: bool = True,
                 **kwargs):
        super(BiMPMatching, self).__init__(**kwargs)
        self.supports_masking = True
        self.num_perspectives = num_perspectives
        self.share_weights_between_directions = share_weights_between_directions

        if with_full_match and is_forward is None:
            raise ValueError("Must specify is_forward to enable full matching")
        self.is_forward = is_forward

        self.with_full_match = with_full_match
        self.with_maxpool_match = with_maxpool_match
        self.with_attentive_match = with_attentive_match
        self.with_max_attentive_match = with_max_attentive_match

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        hidden_size = int(input_shape[-1] / 2)

        if self.with_full_match:
            self.full_match_weight = self._create_weight(
                (self.num_perspectives, hidden_size), "full_match_weight")
            self.full_match_weight_reversed = self._create_reversed_weight(
                self.full_match_weight, "full_match_weight_reversed")

        if self.with_maxpool_match:
            self.maxpool_match_weight = self._create_weight(
                (self.num_perspectives, hidden_size), "maxpool_match_weight")

        if self.with_attentive_match:
            self.attentive_match_weight = self._create_weight(
                (self.num_perspectives, hidden_size), "attentive_match_weight")
            self.attentive_match_weight_reversed = self._create_reversed_weight(
                self.attentive_match_weight, "attentive_match_weight_reversed")

        if self.with_max_attentive_match:
            self.max_attentive_match_weight = self._create_weight(
                (self.num_perspectives, hidden_size),
                "max_attentive_match_weight")
            self.max_attentive_match_weight_reversed = self._create_reversed_weight(
                self.max_attentive_match_weight,
                "max_attentive_match_weight_reversed")

    def _create_weight(self, shape, name):
        weights = self.add_weight(shape=shape, initializer="glorot_uniform",
                                  trainable=True, name=name)
        return weights

    def _create_reversed_weight(self, weight, name):
        return weight if self.share_weights_between_directions \
            else self._create_weight(K.int_shape(weight), name)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        return (input_shape[0], input_shape[1], self.num_perspectives * 2 * self.num_strategies)

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = {
            'num_perspectives': self.num_perspectives,
            'share_weights_between_directions': self.share_weights_between_directions,
            'is_forward': self.is_forward,
            'with_full_match': self.with_full_match,
            'with_maxpool_match': self.with_maxpool_match,
            'with_attentive_match': self.with_attentive_match,
            'with_max_attentive_match': self.with_max_attentive_match}
        base_config = super(BiMPMatching, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, mask=None):
        context_1, context_2 = inputs
        shape_1, shape_2 = K.int_shape(context_1), K.int_shape(context_2)
        hidden_size = int(shape_1[-1] / 2)

        if self.is_forward:
            context_1 = context_1[:, :, :hidden_size]
            context_2 = context_2[:, :, :hidden_size]
        else:
            context_1 = context_1[:, :, hidden_size:]
            context_2 = context_2[:, :, hidden_size:]

        # # mask: (batch,)
        if mask is None:
            mask_1 = K.cast(K.ones(shape_1[:-1]), 'bool')
            mask_2 = K.cast(K.ones(shape_2[:-1]), 'bool')
        else:
            mask_1, mask_2 = mask

        matching_vec_1, matching_vec_2 = [], []

        # Step 0. unweighted cosine
        # First calculate the cosine similarities between each forward
        # (or backward) contextual embedding and every forward (or backward)
        # contextual embedding of the other sentence

        # (batch, timesteps_1, timesteps_2)
        cosine_sim = cosine_matrix(context_1, context_2)

        # (batch, seq_len*, 1)
        cosine_max_1 = masked_max(cosine_sim, K.expand_dims(mask_2, axis=-2),
                                  axis=2, keepdims=True)
        cosine_mean_1 = masked_mean(cosine_sim, K.expand_dims(mask_2, axis=-2),
                                    axis=2, keepdims=True)
        permuted_cosine_sim = K.permute_dimensions(cosine_sim, (0, 2, 1))
        cosine_max_2 = masked_max(permuted_cosine_sim,
                                  K.expand_dims(mask_1, axis=-2),
                                  axis=2, keepdims=True)
        cosine_mean_2 = masked_mean(permuted_cosine_sim,
                                    K.expand_dims(mask_1, axis=-2),
                                    axis=2, keepdims=True)
        matching_vec_1.extend([cosine_max_1, cosine_mean_1])
        matching_vec_2.extend([cosine_max_2, cosine_mean_2])

        # Step 1. Full-Matching
        # Each time step of forward (or backward) contextual embedding of one sentence
        # is compared with the last time step of the forward (or backward)
        # contextual embedding of the other sentence
        if self.with_full_match:
            if self.is_forward:
                context_1_last = K.expand_dims(
                    get_last_value(context_1, mask_1), axis=1)
                context_2_last = K.expand_dims(
                    get_last_value(context_2, mask_2), axis=1)
            else:
                context_1_last = context_1[:, 0:1, :]
                context_2_last = context_2[:, 0:1, :]
            matching_vec_1_full = multi_perspective_match(
                context_1, context_2_last, self.full_match_weight)
            matching_vec_2_full = multi_perspective_match(
                context_2, context_1_last, self.full_match_weight_reversed)
            matching_vec_1.extend(matching_vec_1_full)
            matching_vec_2.extend(matching_vec_2_full)

        # Step 2. Maxpooling-Matching
        # Each time step of forward (or backward) contextual embedding of one sentence
        # is compared with every time step of the forward (or backward)
        # contextual embedding of the other sentence, and only the max value of each
        # dimension is retained.
        if self.with_maxpool_match:
            # (batch, seq_len1, seq_len2, num_perspectives)
            matching_vector_max = multi_perspective_match_pairwise(
                context_1, context_2, self.maxpool_match_weight)

            # (batch, seq_len*, num_perspectives)
            matching_vector_1_max = masked_max(
                matching_vector_max,
                K.expand_dims(K.expand_dims(mask_2, axis=-2), axis=-1),
                axis=2)
            matching_vector_1_mean = masked_mean(
                matching_vector_max,
                K.expand_dims(K.expand_dims(mask_2, axis=-2), axis=-1),
                axis=2)
            permuted_matching_vector_max = K.permute_dimensions(
                matching_vector_max, (0, 2, 1, 3))
            matching_vector_2_max = masked_max(
                permuted_matching_vector_max,
                K.expand_dims(K.expand_dims(mask_1, axis=-2), axis=-1),
                axis=2)
            matching_vector_2_mean = masked_mean(
                permuted_matching_vector_max,
                K.expand_dims(K.expand_dims(mask_1, axis=-2), axis=-1),
                axis=2)
            matching_vec_1.extend(
                [matching_vector_1_max, matching_vector_1_mean])
            matching_vec_2.extend(
                [matching_vector_2_max, matching_vector_2_mean])

        # Step 3. Attentive-Matching
        # Each forward (or backward) similarity is taken as the weight
        # of the forward (or backward) contextual embedding, and calculate an
        # attentive vector for the sentence by weighted summing all its
        # contextual embeddings.
        # Finally match each forward (or backward) contextual embedding
        # with its corresponding attentive vector.

        # (batch, seq_len1, seq_len2, hidden_size)
        att_2 = tf.expand_dims(context_2, -3) * tf.expand_dims(cosine_sim, -1)
        att_1 = tf.expand_dims(context_1, -2) * tf.expand_dims(cosine_sim, -1)

        if self.with_attentive_match:
            # (batch, seq_len*, hidden_size)
            att_mean_2 = masked_softmax(tf.reduce_sum(att_2, axis=2),
                                        tf.expand_dims(mask_1, axis=-1))
            att_mean_1 = masked_softmax(tf.reduce_sum(att_1, axis=1),
                                        tf.expand_dims(mask_2, axis=-1))

            # (batch, seq_len*, num_perspectives)
            matching_vec_1_att_mean = multi_perspective_match(
                context_1, att_mean_2, self.attentive_match_weight)
            matching_vec_2_att_mean = multi_perspective_match(
                context_2, att_mean_1, self.attentive_match_weight_reversed)
            matching_vec_1.extend(matching_vec_1_att_mean)
            matching_vec_2.extend(matching_vec_2_att_mean)

        # Step 4. Max-Attentive-Matching
        # Pick the contextual embeddings with the highest cosine similarity as the attentive
        # vector, and match each forward (or backward) contextual embedding with its
        # corresponding attentive vector.
        if self.with_max_attentive_match:
            # (batch, seq_len*, hidden_size)
            att_max_2 = masked_max(
                att_2,
                tf.expand_dims(tf.expand_dims(mask_2, -2), -1),
                axis=2)
            att_max_1 = masked_max(
                tf.transpose(att_1, (0, 2, 1, 3)),
                tf.expand_dims(tf.expand_dims(mask_1, -2), -1),
                axis=2)

            # (batch, seq_len*, num_perspectives)
            matching_vec_1_att_max = multi_perspective_match(
                context_1, att_max_2, self.max_attentive_match_weight)
            matching_vec_2_att_max = multi_perspective_match(
                context_2, att_max_1, self.max_attentive_match_weight_reversed)
            matching_vec_1.extend(matching_vec_1_att_max)
            matching_vec_2.extend(matching_vec_2_att_max)

        matching_tensor_1 = K.concatenate(matching_vec_1)
        matching_tensor_2 = K.concatenate(matching_vec_2)
        return matching_tensor_1, matching_tensor_2


def multi_perspective_match(tensor_1, tensor_2, weight) \
        -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Calculate multi-perspective cosine matching between time-steps of vectors
    of the same length.

    Parameters
    ----------
    tensor_1 : ``tf.Tensor``
        A tensor of shape ``(batch, seq_len, hidden_size)``
    tensor_2 : ``tf.Tensor``
        A tensor of shape ``(batch, seq_len or 1, hidden_size)``
    weight : ``tf.Tensor``
        A tensor of shape ``(num_perspectives, hidden_size)``

    Returns
    -------
    A tuple of two tensors consisting multi-perspective matching results.
    The first one is of the shape (batch, seq_len, 1), the second one is of shape
    (batch, seq_len, num_perspectives)
    """
    assert K.int_shape(tensor_1)[0] == K.int_shape(tensor_2)[0]
    assert K.int_shape(weight)[1] == K.int_shape(tensor_1)[2] \
        == K.int_shape(tensor_2)[2]

    # (batch, seq_len, 1)
    similarity_single = K.expand_dims(cosine_similarity(tensor_1, tensor_2),
                                      axis=-1)

    # (1, 1, num_perspectives, hidden_size)
    weight = K.expand_dims(K.expand_dims(weight, axis=0), axis=0)

    # (batch, seq_len, num_perspectives, hidden_size)
    tensor_1 = weight * K.expand_dims(tensor_1, axis=2)
    tensor_2 = weight * K.expand_dims(tensor_2, axis=2)

    similarity_multi = cosine_similarity(tensor_1, tensor_2)

    return similarity_single, similarity_multi


def multi_perspective_match_pairwise(tensor_1: tf.Tensor, tensor_2: tf.Tensor,
                                     weight: tf.Tensor) -> tf.Tensor:
    """
    Calculate multi-perspective cosine matching between each time step of
    one vector and each time step of another vector.

    Parameters
    ----------
    tensor_1 : ``tf.Tensor``
        A tensor of shape ``(batch, seq_len1, hidden_size)``
    tensor_2 : ``tf.Tensor``
        A tensor of shape ``(batch, seq_len2, hidden_size)``
    weight : ``torch.Tensor``
        A tensor of shape ``(num_perspectives, hidden_size)``

    Returns
    -------
    A tensor of shape (batch, seq_len1, seq_len2, num_perspectives) consisting
    multi-perspective matching results
    """
    num_perspectives = K.int_shape(weight)[0]

    # (1, num_perspectives, 1, hidden_size)
    weight = tf.expand_dims(tf.expand_dims(weight, axis=0), axis=2)

    # (batch, num_perspectives, seq_len*, hidden_size)
    tensor_1 = weight * tf.tile(tf.expand_dims(tensor_1, 1),
                                multiples=(1, num_perspectives, 1, 1))
    tensor_2 = weight * tf.tile(tf.expand_dims(tensor_2, 1),
                                multiples=(1, num_perspectives, 1, 1))

    # (batch, num_perspectives, seq_len*, 1)
    tensor_1_norm = tf.norm(tensor_1, ord=2, axis=3, keepdims=True)
    tensor_2_norm = tf.norm(tensor_2, ord=2, axis=3, keepdims=True)

    # (batch, num_perspectives, seq_len1, seq_len2)
    mul_result = tf.matmul(tensor_1,
                           tf.transpose(tensor_2, (0, 1, 3, 2)))
    norm_value = tensor_1_norm * tf.transpose(tensor_2_norm, (0, 1, 3, 2))

    # (batch, seq_len1, seq_len2, num_perspectives)
    res = mul_result / (norm_value + K.epsilon())
    return K.permute_dimensions(res, (0, 2, 3, 1))
