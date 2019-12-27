# -*- coding: utf-8 -*-

from typing import Dict

from tensorflow.keras.models import Model
import numpy as np

from .predictor import Predictor
from ..data import DataHandler, Tokenizer, JiebaTokenizer, Example


class TextClassifierPredictor(Predictor):

    def __init__(self, model: Model, data_handler: DataHandler,
                 tokenizer: Tokenizer = None):
        super(TextClassifierPredictor, self).__init__(model, data_handler)
        self.tokenizer = tokenizer if tokenizer else JiebaTokenizer()

    def _json_to_example(self, json_dict: Dict) -> Example:
        if isinstance(json_dict['text'], str):
            tokens = [t.word for t in self.tokenizer.tokenize(
                json_dict['text'])]
        elif isinstance(json_dict, list):
            tokens = json_dict['text']
        else:
            raise ValueError("Invalid text type: %s" % type(json_dict['text']))
        return self.data_handler.make_example(tokens)

    def decode(self, preds: np.ndarray) -> Dict[str, np.ndarray]:
        classes = [self.targets[self.label_field].vocab.itos[idx]
                   for idx in np.argmax(preds, axis=-1)]
        return {self.label_field: classes}
