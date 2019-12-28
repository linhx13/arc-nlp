# -*- coding: utf-8 -*-

from .crf_accuracies import crf_accuracy, crf_marginal_accuracy
from .crf_accuracies import crf_viterbi_accuracy


def get_module_objects():
    return dict(globals())
