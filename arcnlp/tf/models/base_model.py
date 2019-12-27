# -*- coding: utf-8 -*-

import logging
import os
import pickle
from typing import Dict, List, Union

import tensorflow as tf
import numpy as np

from arcnlp_tf.data import Field
from arcnlp_tf.layers import seq2vec_encoders

logger = logging.getLogger(__name__)


MODEL_FILE = "model.h5"
CHECKPOINT_MODEL_FILE = "model.epoch_{epoch:d}.h5"
DATA_HANDLER_FILE = "data_handler.pkl"
CUSTOM_OBJECTS_FILE = "custom_objects.pkl"


class BaseModel(object):

    def __init__(self, features: Dict[str, Field], targets: Dict[str, Field]):
        self.features = features
        self.targets = targets
        self.tf_model = None

    def build_model(self):
        inputs = {n: tf.keras.layers.Input(shape=(f.fix_length,), name=n)
                  for n, f in self.features.items()}
        outputs = self.call(**inputs)
        if isinstance(outputs, dict):
            outputs = list(outputs.values())
        self.tf_model = tf.keras.models.Model(inputs=list(inputs.values()),
                                              outputs=outputs)

    def call(self, inputs):
        raise NotImplementedError

    def decode(self, preds: np.ndarray) -> Dict[str, np.ndarray]:
        return {'preds': preds}

    @classmethod
    def get_custom_objects(cls):
        return {}

    # def _get_text_input(self, inputs: List[tf.Tensor], name) \
    #         -> Dict[str, tf.Tensor]:
    #     res = {}
    #     for tensor in inputs:
    #         arr = tensor._keras_history[0].name.split('.', 1)
    #         if arr[0] == name:
    #             res[arr[1]] = tensor
    #     return res

    def _get_text_input(self, inputs: Dict[str, tf.Tensor], name) \
            -> Dict[str, tf.Tensor]:
        res = {}
        for n, t in inputs.items():
            arr = n.split('.', 1)
            if arr[0] == name:
                res[arr[1]] = t
        return res

    # def _get_input(self, inputs: List[tf.Tensor], name) -> tf.Tensor:
    #     for tensor in inputs:
    #         if name == tensor._keras_history[0].name:
    #             return tensor
    #     return None
