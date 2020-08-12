from typing import Iterable, Dict

import tensorflow as tf

from .data_handler import DataHandler
from ..utils import Counter
from ...vocabs import Vocab


class TextMatchingDataHandler(DataHandler):
    def __init__(self, text_field, label_field):
        self.text_field = text_field
        self.label_field = label_field
        super(TextMatchingDataHandler, self).__init__(
            {'premise': text_field, 'hypothesis': text_field},
            {'label': label_field})

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
        example['premise'] = self.text_field.encode(data['premise'])
        example['hypothesis'] = self.text_field.encode(data['hypothesis'])
        if data.get('label') is not None:
            example['label'] = self.label_field.encode(data['label'])
        return example

    def build_vocab(self, *args, **kwargs):
        text_counter, label_counter = Counter(), Counter()
        for examples in args:
            for ex in examples:
                self.text_field.count_vocab(ex['premise'], text_counter)
                self.text_field.count_vocab(ex['hypothesis'], text_counter)
                self.label_field.count_vocab(ex['label'], label_counter)
        self.text_field.vocab = Vocab(text_counter)
        self.label_field.vocab = Vocab(label_counter, unknown_token=None,
                                       reserved_tokens=[])

    def element_length_func(self, example) -> int:
        return tf.shape(example['premise'])[0]
