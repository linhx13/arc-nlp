# -*- coding: utf-8 -*-

import logging
from typing import Dict, Iterable
import csv

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from .data_handler import DataHandler
from .. import Field, LabelField, Example
from ..tokenizers import Tokenizer, WhitespaceTokenizer


logger = logging.getLogger(__name__)


class QuoraQuestionPairsDataHandler(DataHandler):

    def __init__(self,
                 token_fields: Dict[str, Field],
                 tokenizer: Tokenizer = None,
                 sparse_target: bool = True,
                 sort_feature: str = None):
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        label_postprocessing = self._label_sparse if sparse_target \
            else self._label_onehot
        label = LabelField(postprocessing=label_postprocessing)

        super(QuoraQuestionPairsDataHandler, self).__init__(
            {'premise': token_fields, 'hypothesis': token_fields},
            {'label': label}, sort_feature)

    def _label_sparse(self, batch, vocab):
        return np.expand_dims(batch, axis=-1)

    def _label_onehot(self, batch, vocab):
        return tf.keras.utils.to_categorical(batch, len(vocab))

    def _read_from_path(self, path: str) -> Iterable[Dict]:
        logger.info("Reading data from %s" % path)
        with open(path) as fin:
            reader = csv.DictReader(fin)
            for row in tqdm(reader):
                yield self.make_example(premise=row['question1'],
                                        hypothesis=row['question2'],
                                        label=row['is_duplicate'])

    def make_example(self, premise, hypothesis, label=None) -> Example:
        data = {}
        if isinstance(premise, str):
            data['premise'] = [t.text
                               for t in self.tokenizer.tokenize(premise)]
        else:
            data['premise'] = premise
        if isinstance(hypothesis, str):
            data['hypothesis'] = [t.text
                                  for t in self.tokenizer.tokenize(hypothesis)]
        else:
            data['hypothesis'] = self.hypothesis
        if label is not None:
            data['label'] = label
        fields = dict((k, self.example_fields[k]) for k in data)
        return Example.fromdict(data, fields)
