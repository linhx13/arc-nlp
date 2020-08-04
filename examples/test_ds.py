import os
from typing import Dict
import sys
from collections import Counter

import tensorflow as tf

sys.path.append("../")
import arcnlp.tf


train_path = os.path.expanduser("~/datasets/LCQMC/train.txt")
print(train_path)


# class Field(arcnlp.tf.data.Field):

#     def __init__(self, **kwargs):
#         super(Field, self).__init__(**kwargs)


# class Example(object):

#     @classmethod
#     def fromdict(cls, data, fields):

#         ex = cls()
#         for key, vals in fields.items():
#             if key not in data:
#                 raise ValueError
#             for val in


def tokenizer(text):
    import jieba
    return jieba.lcut(text)


class TextFeature:

    def __init__(self, vocab, tokenizer):
        self.vocab = vocab
        self.tokenizer = tokenizer

    def preprocess(self, x):
        if isinstance(x, tf.Tensor):
            x = tf.compat.as_text(x.numpy())
        return self.tokenizer(x)

    def encode(self, x):
        pass


class Label:
    def __init__(self, vocab):
        self.vocab = vocab

    def preprocess(self, x):
        if isinstance(x, tf.Tensor):
            x = x.numpy()
        return x


class LCQMC:

    def __init__(self, text_feature, label):
        self.text_feature = text_feature
        self.label = label

    def raw_dataset(self, path) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_generator(
            lambda: self._read(path),
            output_types=self.raw_dataset_types(),
        )
        return ds

    def _read(self, path) -> Dict:
        with open(path) as fin:
            for line in fin:
                line = line.strip("\r\n")
                if not line:
                    continue
                arr = line.split('\t')
                yield {'premise': arr[0], 'hypothesis': arr[1], 'label': int(arr[2])}

    def raw_dataset_types(self):
        return {"premise": tf.string, 'hypothesis': tf.string, 'label': tf.int32}

    def build_example(self, premise, hypothesis, label=None) -> Dict:
        example = {}
        example['premise'] = self.text_feature.preprocess(premise)
        example['hypothesis'] = self.text_feature.preprocess(hypothesis)
        if label is not None:
            example['label'] = self.label.preprocess(label)
        return example

    def build_vocab(self, raw_ds: tf.data.Dataset):
        token_counter = Counter()
        for ex in raw_ds:
            print(ex)
            for t in self.premise.preprocess(ex['premise']):
                token_counter[t] += 1
            for t in self.hypothesis.preprocess(ex['hypothesis']):
                token_counter[t] += 1
        print(token_counter)
        self.premise.vocab = arcnlp.tf.data.Vocab(token_counter)
        self.hypothesis


    def build_dataset(self, dataset) -> tf.data.Dataset:
        pass

    def _preprocess_fn(self, example):
        print('_preprocess_fn:', example)
        # res = {}
        # for k, v in self.features.items():
        #     res[k] = v.preprocess(example[k].numpy())
        # for k, v in self.targets.items():
        #     res[k] = v.preprocess(example[k].numpy())
        # return res
        return example

    def process(self, ds: tf.data.Dataset, batch_size: int) -> tf.data.Dataset:
        for x in ds.take(1):
            print(x)
        ds = ds.map(lambda ex: tf.py_function(func=self._preprocess_fn,
                                         inp=[ex], Tout=self.raw_dataset_types()))
        # ds = ds.window(batch_size)
        return ds



builder = LCQMC(TextFeature(None, tokenizer), Label(None))
raw_ds = builder.raw_dataset(train_path)
# process_ds = builder.process(raw_ds, batch_size=3)

# for idx, ex in enumerate(raw_ds.repeat().take(20)):
#     print('=' * 10)
#     print('idx: %d' % idx)
#     print(ex)
#     print(ex['premise'].numpy().decode('utf-8'))


for idx, ex in enumerate(raw_ds.take(5)):
    print('=' * 10)
    ex = builder.build_example(**ex)
    print(ex)

builder.build_vocab(raw_ds.take(5))

# for batch in raw_ds.window(3).take(2):
#     print('=' * 10)
#     print(batch)
#     print(batch['premise'])
#     for x in batch['premise']:
#         print(x.numpy().decode('utf-8'))
#     print(batch['hypothesis'])
#     for x in batch['hypothesis']:
#         print(x.numpy().decode('utf-8'))
#     print(batch['label'])
#     for x in batch['label']:
#         print(x.numpy())


# for example in process_ds.take(5):
#     print('=' * 10)
#     print(example)
