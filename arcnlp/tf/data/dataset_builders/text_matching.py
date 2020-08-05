from typing import Iterable, Dict

import tensorflow as tf

from .dataset_builder import DatasetBuilder
from ..utils import Counter
from ...vocab import Vocab


class TextMatchingData(DatasetBuilder):
    def __init__(self, text_feature, label):
        self.text_feature = text_feature
        self.label = label
        super(TextMatchingData, self).__init__(
            {'premise': text_feature, 'hypothesis': text_feature},
            {'label': label})

    def read_from_path(self, path) -> Iterable[Dict]:
        with open(path) as fin:
            for line in fin:
                line = line.strip("\r\n")
                if not line:
                    continue
                arr = line.split('\t')
                yield self.build_example(premise=arr[0].split(),
                                         hypothesis=arr[1].split(),
                                         label=arr[2])

    def build_example(self, premise, hypothesis, label=None) -> Dict:
        example = {}
        example['premise'] = self.text_feature.preprocess(premise)
        example['hypothesis'] = self.text_feature.preprocess(hypothesis)
        if label is not None:
            example['label'] = self.label.preprocess(label)
        return example

    def build_vocab(self, *examples):
        text_counter, label_counter = Counter(), Counter()
        for raw_data in examples:
            for ex in (raw_data):
                text_counter.update(ex['premise'])
                text_counter.update(ex['hypothesis'])
                label_counter[ex['label']] += 1
        self.text_feature.vocab = Vocab(text_counter)
        self.label.vocab = Vocab(label_counter, unknown_token=None)

    def element_length_func(self, example) -> int:
        return tf.shape(example['premise'])[0]
