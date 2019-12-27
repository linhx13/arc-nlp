# -*- coding: utf-8 -*-

from tensorflow.keras.models import Model

from . import SequenceTaggerPredictor
from ..data import NerDataHanlder


class NerPredictor(SequenceTaggerPredictor):

    def __init__(self, model: Model,
                 data_handler: NerDataHanlder):
        if data_handler.use_seg_feature:
            featurizers = {
                'seg': data_handler.get_seg_seq
            }
        else:
            featurizers = {}
        super(NerPredictor, self).__init__(model, data_handler,
                                           featurizers)
