# -*- coding: utf-8 -*-

from collections import OrderedDict

from .bow_encoder import BOWEncoder
from .cnn_encoder import CNNEncoder

encoders = OrderedDict()
encoders["bow"] = BOWEncoder
encoders["cnn"] = CNNEncoder
