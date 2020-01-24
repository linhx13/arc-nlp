# -*- coding: utf-8 -*-

from typing import Dict, Iterable

import tensorflow as tf

from .data_handler import DataHandler
from .. import Field, LabelField, Example
from ..tokenizers import Tokenizer, JiebaTokenizer


class FasttextDataHandler(DataHandler):

    def __init__(self,
                 token_fields: Dict[str, Field],
                 tokenizer: Tokenizer = None,
                 label_prefix: str = '__label__',
                 **kwargs):
        self.tokenizer = tokenizer or JiebaTokenizer()
        self.label_prefix = label_prefix
        label = LabelField(postprocessing=self._label_postprocessing)
        super(FasttextDataHandler, self).__init__(
            feature_fields={'tokens': token_fields},
            target_fields={'label': label},
            **kwargs)

    def _label_postprocessing(self, batch, vocab):
        return tf.keras.utils.to_categorical(batch, len(vocab))

    def read_from_path(self, path: str) -> Iterable[Dict]:
        with open(path, errors='ignore') as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                tokens, label = [], None
                for word in line.split():
                    if word.startswith(self.label_prefix):
                        label = word
                    else:
                        tokens.append(word)
                if tokens and label:
                    yield self.build_example(tokens, label)

    def build_example(self, text, label: str = None) -> Example:
        data = {}
        if isinstance(text, str):
            data['tokens'] = [t.text for t in self.tokenizer.tokenize(text)]
        else:
            data['tokens'] = text
        if label is not None:
            data['label'] = label
        fields = dict((k, self.example_fields[k]) for k in data)
        return Example.fromdict(data, fields)
