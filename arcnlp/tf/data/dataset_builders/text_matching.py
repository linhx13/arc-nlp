from typing import Iterable, Dict

import tensorflow as tf

from .dataset_builder import DatasetBuilder
from ..utils import Counter
from ...vocabs import Vocab


class TextMatchingData(DatasetBuilder):
    def __init__(self, text_feature, label):
        self.text_feature = text_feature
        self.label = label
        super(TextMatchingData, self).__init__(
            {'premise': text_feature, 'hypothesis': text_feature},
            {'label': label})

    def read_examples(self, path) -> Iterable[Dict]:
        with open(path) as fin:
            for line in fin:
                line = line.strip("\r\n")
                if not line:
                    continue
                arr = line.split('\t')
                yield {'premise': arr[0].split(),
                       'hypothesis': arr[1].split(),
                       'label': arr[2]}

    def encode_example(self, data: Dict) -> Dict:
        example = {}
        example['premise'] = self.text_feature.encode(data['premise'])
        example['hypothesis'] = self.text_feature.encode(data['hypothesis'])
        if data.get('label') is not None:
            example['label'] = self.label.encode(data['label'])
        return example

    def build_vocab(self, *args, **kwargs):
        text_counter, label_counter = Counter(), Counter()
        for examples in args:
            for ex in examples:
                self.text_feature.count_vocab(ex['premise'], text_counter)
                self.text_feature.count_vocab(ex['hypothesis'], text_counter)
                self.label.count_vocab(ex['label'], label_counter)
        self.text_feature.vocab = Vocab(text_counter)
        self.label.vocab = Vocab(label_counter, unknown_token=None,
                                 reserved_tokens=[])

    def element_length_func(self, example) -> int:
        return tf.shape(example['premise'])[0]
