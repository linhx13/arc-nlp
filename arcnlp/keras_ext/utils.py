# -*- coding: utf-8 -*-

import os
import errno

import tensorflow as tf
import numpy as np


def auto_select_gpu(top_n=1):
    cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Free'
    gpu_memory = [int(x.split()[2]) for x in os.popen(cmd).readlines()]
    if gpu_memory:
        gpu_devices = np.argsort(gpu_memory)
    return ','.join(map(str, gpu_devices[-top_n:]))


def create_tf_session(allow_soft_placement=True,
                      per_process_gpu_memory_fraction=0.5,
                      gpu_allow_growth=True,
                      visible_device_list=None,
                      **kwargs):
    config = tf.ConfigProto()
    config.allow_soft_placement = allow_soft_placement
    config.gpu_options.per_process_gpu_memory_fraction = \
        per_process_gpu_memory_fraction
    config.gpu_options.allow_growth = gpu_allow_growth
    assert visible_device_list is None or isinstance(visible_device_list, str)
    if visible_device_list is not None:
        config.gpu_options.visible_device_list = visible_device_list
    else:
        config.gpu_options.visible_device_list = auto_select_gpu()
    return tf.Session(config=config)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
