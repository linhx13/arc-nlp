# -*- coding: utf-8 -*-

from typing import Dict, Iterable
import json

import tensorflow as tf

from .data_handler import DataHandler
from .. import Field, LabelField, Example
from ..tokenizers import Tokenizer, JiebaTokenizer


class TextClassficationJsonDataHandler(DataHandler):

    def __init__(self,
                 token_fields: Dict[str, Field],
                 tokenizer: Tokenizer = None,
                 sort_feature: str = None):
        self.tokenizer = tokenizer or JiebaTokenizer()
        label = LabelField(postprocessing=self._label_postprocessing)
        super(TextClassficationJsonDataHandler, self).__init__(
            feature_fields={'tokens': token_fields},
            target_fields={'label': label},
            sort_feature=sort_feature)

    def _label_postprocessing(self, batch, vocab):
        return tf.keras.utils.to_categorical(batch, len(vocab))

    def _read_from_path(self, path: str) -> Iterable[Dict]:
        with open(path, errors='ignore') as fin:
            for line in fin:
                data = json.loads(line)
                yield self.make_example(data['text'], data['label'])

    def make_example(self, text, label=None) -> Example:
        data = {}
        if isinstance(text, str):
            data['tokens'] = [t.text for t in self.tokenizer.tokenize(text)]
        else:
            data['tokens'] = text
        if label is not None:
            data['label'] = label
        fields = dict((k, self.example_fields[k]) for k in data)
        return Example.fromdict(data, fields)
