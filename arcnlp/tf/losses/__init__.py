# -*- coding: utf-8 -*-

from .crf_losses import crf_loss, crf_nll
from .amsoftmax import amsoftmax_loss, sparse_amsoftmax_loss


def get_module_objects():
    return dict(globals())