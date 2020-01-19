# -*- coding: utf-8 -*-

from typing import Dict, Iterable
import logging

import tensorflow as tf

from .data_handler import DataHandler
from .. import Field, LabelField, Example
from ..tokenizers import Tokenizer, JiebaTokenizer

logger = logging.getLogger(__name__)


class TextMatchingDataHandler(DataHandler):

    def __init__(self,
                 token_fields: Dict[str, Field],
                 tokenizer: Tokenizer = None,
                 sort_feature: str = None):
        self.tokenizer = tokenizer or JiebaTokenizer()
        label = LabelField(postprocessing=self._label_postprocessing)
        super(TextMatchingDataHandler, self).__init__(
            {'premise': token_fields, 'hypothesis': token_fields},
            {'label': label}, sort_feature)

    def _label_postprocessing(self, batch, vocab):
        return tf.keras.utils.to_categorical(batch, len(vocab))

    def _read_from_path(self, path: str) -> Iterable[Dict]:
        logger.info("Reading data from %s" % path)
        with open(path) as fin:
            for line in fin:
                line = line.strip('\r\n')
                arr = line.split('\t')
                yield self.build_example(premise=arr[0].split(),
                                         hypothesis=arr[1].split(),
                                         label=arr[2])

    def build_example(self, premise, hypothesis, label=None) -> Example:
        data = {}
        if isinstance(premise, str):
            data['premise'] = [
                t.text for t in self.tokenizer.tokenize(premise)]
        else:
            data['premise'] = premise
        if isinstance(hypothesis, str):
            data['hypothesis'] = [
                t. text for t in self.tokenzier.tokenize(hypothesis)]
        else:
            data['hypothesis'] = hypothesis
        if label is not None:
            data['label'] = label
        fields = dict((k, self.example_fields[k]) for k in data)
        return Example.fromdict(data, fields)
