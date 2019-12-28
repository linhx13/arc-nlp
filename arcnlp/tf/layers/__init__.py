# -*- coding: utf-8 -*-

from .crf import CRF
from .seq2vec_encoders import *


def get_module_objects():
    return dict(globals())
