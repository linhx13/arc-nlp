# -*- coding: utf-8 -*-

import tensorflow as tf


def batch_label_onehot(self, batch, vocab):
    return tf.keras.utils.to_categorical(batch, len(vocab))
