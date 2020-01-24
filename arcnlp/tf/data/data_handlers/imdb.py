# -*- coding: utf-8 -*-

from typing import Iterable, Dict
import os
import glob
import logging

import tensorflow as tf

from .data_handler import DataHandler
from .. import Example, Field, LabelField


logger = logging.getLogger(__name__)


class IMDBDataHandler(DataHandler):

    def __init__(self,
                 token_fields: Dict[str, Field],
                 sort_feature: str = None):
        label = LabelField(postprocessing=self._label_postprocessing)
        super(IMDBDataHandler, self).__init__(
            feature_fields={'tokens': token_fields},
            target_fields={'label': label},
            sort_feature=sort_feature)

    def _label_postprocessing(self, batch, vocab):
        return tf.keras.utils.to_categorical(batch, len(vocab))

    def read_from_path(self, path: str) -> Iterable[Example]:
        assert os.path.isdir(path), "path %s doesn't exist" % path
        for label in ['pos', 'neg']:
            logger.info("Reading instances from path %s" %
                        os.path.join(path, label))
            for fname in glob.iglob(os.path.join(path, label, '*.txt')):
                with open(fname, 'r', encoding='utf-8') as f:
                    text = f.readline()
                    yield self.build_example(text, label)

    def build_example(self, text, label=None) -> Example:
        data = {}
        if isinstance(text, str):
            data['tokens'] = text.split()
        else:
            data['tokens'] = text
        if label is not None:
            data['label'] = label
        fields = dict((k, self.example_fields[k]) for k in data)
        return Example.fromdict(data, fields)
