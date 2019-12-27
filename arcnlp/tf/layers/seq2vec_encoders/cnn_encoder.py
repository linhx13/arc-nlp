# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import InputSpec, Conv1D, Concatenate, Dense
from tensorflow.keras.regularizers import l1_l2


class CNNEncoder(tf.keras.layers.Layer):
    """ CNNEncoder is a combination of multiple convolutional layers and max
    pooling layers. This is defined as a single layer to be consistent with
    other encoders in terms of input and output specifications.

    Input shape: (batch_size, sequence_length, input_dim).
    Output shape: (batch_size, output_dim).

    The CNN has one convolution layer per each ngram filter size. Each
    convolution operation gives out a vector of size num_filters. The number
    of times a convolution layer will be used depends on the ngram size:
    input_len - ngram_size + 1. The corresponding maxpooling layer aggregates
    all these outputs from the convolution layer and outputs the max.

    This operation is repeated for every ngram size passed, and consequently
    the dimensionality of the output after maxpooling is
    len(ngram_filter_sizes) * num_filters.

    We the use a fully connected layer to project in back to the desired
    output_dim.

    References: "A Sensitivity Analysis of (and Practitioners’ Guide to)
    Convolutional Neural Networks for Sentence Classification",
    Zhang and Wallace 2016, particularly Figure 1.

    Args:
        filters: Integer, the output dim for each convolutional layer.
        kernel_sizes: An integer tuple of list, the kernel sizes of each
            convolutional layers.
        units: After doing convolutions, we'll project the collected features
            into a vecor of this size. If this value is `None`, just return the
            result of the max pooling.
        conv_layer_activation: string of convolutional layer `Activation`.
        l1_regularization: float.
        l2_regularization: float.
    """

    def __init__(self, filters=100, kernel_sizes=(2, 3, 4, 5),
                 conv_layer_activation='relu',
                 l1_regularization=None, l2_regularization=None,
                 units=None,
                 **kwargs):
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.units = units
        self.conv_layer_activation = conv_layer_activation
        self.l1_regularization = l1_regularization
        self.l2_regularization = l2_regularization
        self.regularizer = l1_l2(
            l1=l1_regularization if l1_regularization is not None else 0.0,
            l2=l2_regularization if l2_regularization is not None else 0.0)
        self.conv_layers = None
        self.projection_layer = None
        self.trainable_layers = None
        self.output_dim = None

        self.input_spec = [InputSpec(ndim=3)]
        super(CNNEncoder, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv_layers = [Conv1D(filters=self.filters,
                                   kernel_size=kernel_size,
                                   activation=self.conv_layer_activation,
                                   kernel_regularizer=self.regularizer,
                                   bias_regularizer=self.regularizer)
                            for kernel_size in self.kernel_sizes]
        for conv_layer in self.conv_layers:
            with K.name_scope(conv_layer.name):
                conv_layer.build(input_shape)
        maxpool_output_dim = self.filters * len(self.kernel_sizes)
        if self.units is not None:
            self.projection_layer = Dense(self.units)
            projection_input_shape = (input_shape[0], maxpool_output_dim)
            with K.name_scope(self.projection_layer.name):
                self.projection_layer.build(projection_input_shape)
            self.output_dim = self.units
            self.trainable_layers = self.conv_layers + [self.projection_layer]
        else:
            self.projection_layer = None
            self.output_dim = maxpool_output_dim
            self.trainable_layers = self.conv_layers

        super(CNNEncoder, self).build(input_shape)

    def call(self, inputs, mask=None):
        # Each convolution layer returns output of size (batch_size, conv_length, filters),
        # where `conv_length = num_words - kernel_size + 1`. We then do max
        # pooling over each filter for the whole input sequence, just use K.max,
        # giving a result tensor of shape (batch_size, filters), which then
        # gets projected using the projection layer.
        filter_outputs = [K.max(conv_layer.call(inputs), axis=1)
                          for conv_layer in self.conv_layers]
        maxpool_output = Concatenate()(filter_outputs) \
            if len(filter_outputs) > 1 else filter_outputs[0]
        if self.projection_layer:
            result = self.projection_layer.call(maxpool_output)
        else:
            result = maxpool_output
        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def compute_mask(self, inputs, mask=None):
        # By default Keras propagates the mask from a layer that supports masking. We don't need it
        # anymore. So eliminating it from the flow.
        return None

    def get_config(self):
        config = {"filters": self.filters,
                  "kernel_sizes": self.kernel_sizes,
                  "units": self.units,
                  "conv_layer_activation": self.conv_layer_activation,
                  "l1_regularization": self.l1_regularization,
                  "l2_regularization": self.l2_regularization
                  }
        base_config = super(CNNEncoder, self).get_config()
        config.update(base_config)
        return config

    @property
    def trainable_weights(self):
        trainable_weights = []
        for layer in self.trainable_layers:
            trainable_weights.extend(layer.trainable_weights)
        return trainable_weights
