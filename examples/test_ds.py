import os
from typing import Dict, List, Union, Iterable
import sys
from collections import Counter, defaultdict

import tensorflow as tf

sys.path.append("../")
import arcnlp.tf


train_path = os.path.expanduser("~/datasets/LCQMC/train_seg.txt")
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


class Vocab:

    def __init__(self, counter):
        sorted_tokens = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self._idx_to_token = [x[0] for x in sorted_tokens]
        self._token_to_idx = defaultdict()
        self._token_to_idx.update({tok: idx for idx, tok in enumerate(self._idx_to_token)})

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self._token_to_idx[tokens]
        else:
            return [self._token_to_idx[token] for token in tokens]

    def __len__(self):
        return len(self._idx_to_token)

    def __call__(self, tokens):
        return self[tokens]


class TextFeature:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab = None

    def __call__(self, x) -> List[Union[str, int]]:
        if isinstance(x, tf.Tensor):
            x = x.numpy()
        if isinstance(x, list):
            x = [tf.compat.as_text(t) for t in x]
        elif isinstance(x, str):
            x = tf.compat.as_text(x)
            x = self.tokenizer(x)
        if self.vocab:
            x = self.vocab(x)
        return x


class Label:
    def __init__(self):
        self.vocab = None

    def __call__(self, x) -> Union[str, int]:
        if isinstance(x, tf.Tensor):
            x = x.numpy()
            if isinstance(x, (str, bytes)):
                x = tf.compat.as_text(x)
        if isinstance(x, str) and self.vocab is not None:
            x = self.vocab[x]
        return x


class LCQMC:

    def __init__(self, text_transform, label_transform):
        self.text_transform = text_transform
        self.label_transform = label_transform

    def read_from_path(self, path) -> Iterable[Dict]:
        with open(path) as fin:
            for line in fin:
                line = line.strip("\r\n")
                if not line:
                    continue
                arr = line.split('\t')
                yield {'premise': arr[0].split(),
                       'hypothesis': arr[1].split(),
                       'label': arr[2]}

    def build_example(self, data: Dict) -> Dict:
        example = {}
        example['premise'] = self.text_transform(data['premise'])
        example['hypothesis'] = self.text_transform(data['hypothesis'])
        if data.get('label') is not None:
            example['label'] = self.label_transform(data['label'])
        return example

    def build_vocab(self, *raw_examples):
        text_counter, label_counter = Counter(), Counter()
        for raw_data in raw_examples:
            for ex in (raw_data):
                text_counter.update(self.text_transform(ex['premise']))
                text_counter.update(self.text_transform(ex['hypothesis']))
                label_counter[self.label_transform(ex['label'])] += 1
        print('text_counter:', text_counter)
        print("label_counter:", label_counter)
        self.text_transform.vocab = Vocab(text_counter)
        self.label_transform.vocab = Vocab(label_counter)

    def build_dataset(self, path) -> tf.data.Dataset:
        examples = list(map(self.build_example, self.read_from_path(path)))

        def _gen():
            for ex in examples:
                yield ex

        output_types = {'premise': tf.int32,
                        'hypothesis': tf.int32,
                        'label': tf.int32}

        return tf.data.Dataset.from_generator(_gen, output_types=output_types)

    def build_iter(self, dataset, batch_size=32, train=True) -> tf.data.Dataset:
        padded_shapes = {'premise': [None],
                         'hypothesis': [None],
                         'label': []}
        if train:
            dataset = dataset.shuffle(batch_size * 100)
        data = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def build_bucket_iter(self, dataset, batch_size=32, train=True) -> tf.data.Dataset:
        padded_shapes = {'premise': [None],
                         'hypothesis': [None],
                         'label': []}
        if train:
            bucket_boundaries = self._bucket_boundaries(batch_size)
            bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)
            dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
                self.element_length_func, bucket_boundaries, bucket_batch_sizes,
                padded_shapes=padded_shapes))
            dataset = dataset.shuffle(10)
        else:
            dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def element_length_func(self, example) -> int:
        return tf.shape(example['premise'])[0]

    def _bucket_boundaries(self, max_length, min_length=8, length_bucket_step=1.1):
        boundaries = []
        x = min_length
        while x < max_length:
            boundaries.append(x)
            x = max(x + 1, int(x * length_bucket_step))
        return boundaries


builder = LCQMC(TextFeature(tokenizer), Label())
# train_ds = builder.raw_dataset(train_path)
train_examples = builder.read_from_path(train_path)
builder.build_vocab(train_examples)
train_ds = builder.build_dataset(train_path)

for batch in builder.build_bucket_iter(train_ds).take(3):
    print(batch)

# process_ds = builder.process(raw_ds, batch_size=3)

# for idx, ex in enumerate(raw_ds.repeat().take(20)):
#     print('=' * 10)
#     print('idx: %d' % idx)
#     print(ex)
#     print(ex['premise'].numpy().decode('utf-8'))


# for idx, ex in enumerate(raw_ds.take(5)):
#     print('=' * 10)
#     ex = builder.build_example(ex)
#     print(ex)

# builder.build_vocab(raw_ds.take(5))
# builder.build_dataset(raw_ds.take(5))

# l = [1,2,3,4]

# def gen():
#     with open("./aaa") as fin:
#         for line in fin:
#             yield line.strip()

# ds = tf.data.Dataset.from_generator(gen, output_types=tf.string)
# for x in ds.repeat(2):
    # print(x)
