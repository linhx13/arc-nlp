# -*- coding: utf-8 -*-

import os
import errno

import numpy as np
import tensorflow as tf

from .layers import seq2vec_encoders


def auto_select_gpu(top_n=1):
    cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Free'
    gpu_memory = [int(x.split()[2]) for x in os.popen(cmd).readlines()]
    gpu_devices = np.argsort(gpu_memory) if gpu_memory else []
    return ','.join(map(str, gpu_devices[-top_n:]))


def config_tf_gpu(allow_soft_placement: bool = True,
                  visible_device_list: str = None,
                  top_n_gpu: int = 1,
                  per_process_gpu_memory_fraction: float = 1.0,
                  gpu_allow_growth: bool = True):
    if int(tf.__version__.split('.')[0]) == 1:
        config_tf_v1_gpu(allow_soft_placement,
                         visible_device_list,
                         top_n_gpu,
                         per_process_gpu_memory_fraction,
                         gpu_allow_growth)
    else:
        config_tf_v2_gpu(allow_soft_placement,
                         visible_device_list,
                         top_n_gpu,
                         per_process_gpu_memory_fraction,
                         gpu_allow_growth)


def config_tf_v1_gpu(allow_soft_placement: bool = True,
                     visible_device_list: str = None,
                     top_n_gpu: int = 1,
                     per_process_gpu_memory_fraction: float = 1.0,
                     gpu_allow_growth: bool = True):
    config = tf.ConfigProto()
    config.allow_soft_placement = allow_soft_placement
    if visible_device_list is not None:
        config.gpu_options.visible_device_list = visible_device_list
    else:
        config.gpu_options.visible_device_list = auto_select_gpu(top_n_gpu)
    config.gpu_options.per_process_gpu_memory_fraction = \
        per_process_gpu_memory_fraction
    config.gpu_options.allow_growth = gpu_allow_growth
    tf.keras.backend.set_session(tf.Session(config=config))


def config_tf_v2_gpu(allow_soft_placement: bool = True,
                     visible_device_list: str = None,
                     top_n_gpu: int = 1,
                     per_process_gpu_memory_fraction: float = 1.0,
                     gpu_allow_growth: bool = True):
    tf.config.set_soft_device_placement(allow_soft_placement)
    visible_device_list = visible_device_list or auto_select_gpu(top_n_gpu)
    visible_device_ids = [int(x) for x in visible_device_list.split(',')]
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for i in visible_device_ids:
        tf.config.experimental.set_visible_devices(gpus[i], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], gpu_allow_growth)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_spacy_model(model_name: str, pos_tags: bool, parse: bool, ner: bool):
    import spacy
    disable = ['vectors', 'textcat']
    if not pos_tags:
        disable.append('tagger')
    if not parse:
        disable.append('parser')
    if not ner:
        disable.append('ner')
    return spacy.load(model_name, disable=disable)


def create_span(text, start, end, label):
    return {'text': text, 'start': start, 'end': end, 'label': label}


def bioes_to_spans(text, tags):
    res = []
    cur_label, start = None, None
    for idx, (ch, tag) in enumerate(zip(text, tags)):
        if tag == 'O':
            continue
        prefix, label = tag.split('-', 1)
        if prefix == 'S':
            res.append(create_span(text[idx:idx+1], idx, idx+1, label))
            cur_label, start = None, None
        elif prefix == 'B':
            if cur_label is not None and start is not None:
                res.append(create_span(text[start:idx], start, idx, cur_label))
            cur_label, start = label, idx
        elif prefix == 'I':
            continue
        elif prefix == 'E':
            if cur_label is not None and start is not None:
                res.append(create_span(
                    text[start:idx+1], start, idx+1, cur_label))
            cur_label, start = None, None
    if cur_label is not None and start is not None:
        res.append(create_span(text[start:], start, len(text), cur_label))
    return res


def get_custom_objects():
    custom_objects = {}
    for value in seq2vec_encoders.encoders.values():
        custom_objects[value.__name__] = value
    return custom_objects