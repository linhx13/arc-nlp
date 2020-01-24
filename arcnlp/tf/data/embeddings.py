# -*- coding: utf-8 -*-

import logging
from typing import Union, Dict

import tensorflow as tf
import numpy as np

logger = logging.getLogger(__name__)


def build_embedding_layer(embedding_src: Union[str, Dict[str, np.ndarray]],
                          token_index: Dict[str, int],
                          **kwargs):
    embedding_dim = None
    embedding_matrix = None
    for key, value in _load_embedding_src(embedding_src):
        if embedding_dim is None:
            embedding_dim = value.shape[0]
            embedding_matrix = _init_random_matrix(
                (len(token_index), embedding_dim))
            idx = token_index.get(key)
            if idx is not None:
                embedding_matrix[idx] = value
    return tf.keras.layers.Embedding(len(token_index), embedding_dim,
                                     weights=[embedding_matrix], **kwargs)


def _init_random_matrix(shape, seed=1337):
    np_rng = np.random.RandomState(seed)
    return np_rng.uniform(size=shape, low=0.05, high=-0.05)


def _load_embedding_src(embedding_src):
    if isinstance(embedding_src, dict):
        for key, value in embedding_src.items():
            yield key, value
    elif isinstance(embedding_src, str):
        embedding_dim = None
        with open(embedding_src, errors='ignore') as fin:
            for line in enumerate(fin):
                line = line.strip()
                if not line:
                    continue
                arr = line.split()
                if embedding_dim is None:
                    embedding_dim = int(arr[1])
                else:
                    arr = line.split()
                    if len(arr) - 1 != embedding_dim:
                        logger.warn("Error embedding line: %s" % line)
                        continue
                    value = np.array(list(map(float, arr[1:])))
                    yield arr[0], value
