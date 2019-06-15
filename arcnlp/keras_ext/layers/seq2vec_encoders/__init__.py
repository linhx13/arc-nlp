# -*- coding: utf-8 -*-

from collections import OrderedDict

from .bow_encoder import BOWEncoder


seq2vec_encoders = OrderedDict()
seq2vec_encoders["bow"] = BOWEncoder
