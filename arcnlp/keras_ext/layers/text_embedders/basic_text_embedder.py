# -*- coding: utf-8 -*-

from tensorflow import keras

from .text_field_embedder import TextFieldEmbedder


class BasicTextFieldEmbedder(TextFieldEmbedder):

    def __init__(self, token_embedders):
        """
        Args:
            token_embedders: Dict[str, TokenEmbedder]
        """
        super(BasicTextFieldEmbedder, self).__init__()
        self.token_embedders = token_embedders

    def call(self, inputs):
        """
        Args:
            inputs: Dict[str, tf.Tensor]

        Returns:
            tf.Tensor, the embedded result of the whole text field
        """
        embedder_keys = list(self.token_embedders.keys())
        keys = sorted(embedder_keys)
        embedded = [self.token_embedders[key](inputs[key]) for key in keys]
        print('embedded:', embedded)
        outputs = embedded[0] if len(embedded) == 1 \
            else keras.layers.Concatenate()(embedded)
        print('outputs:', outputs)
        return outputs
