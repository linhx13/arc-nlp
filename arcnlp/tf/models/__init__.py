# -*- coding: utf-8 -*-

from .text_classification import *
from .simple_tagger import SimpleTagger
from .crf_tagger import CrfTagger
# from .malstm import MaLSTM

# _globals = dict(globals())
# model_classes = {}
# for k, v in _globals.items():
#     if type(v) is type and issubclass(v, BaseModel):
#         model_classes[k] = v
