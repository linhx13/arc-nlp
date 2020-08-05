import os
from typing import Dict, List, Union, Iterable
import sys
from collections import Counter, defaultdict

import tensorflow as tf

sys.path.append("../")
import arcnlp.tf


train_path = os.path.expanduser("~/datasets/LCQMC/train_seg.txt")
print(train_path)


def tokenizer(text):
    import jieba
    return jieba.lcut(text)


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
        self.text_transform.vocab = arcnlp.tf.vocab.Vocab(text_counter)
        self.label_transform.vocab = arcnlp.tf.vocab.Vocab(label_counter)

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


# builder = LCQMC(arcnlp.tf.data.features.TextFeature(tokenizer),
#                 arcnlp.tf.data.features.Label())
builder = arcnlp.tf.data.TextMatchingData(
    arcnlp.tf.data.TextFeature(tokenizer),
    arcnlp.tf.data.Label())
# train_ds = builder.raw_dataset(train_path)
train_examples = list(builder.read_from_path(train_path))
for ex in train_examples[:3]:
    print(ex)
builder.build_vocab(train_examples)
train_ds = builder.build_dataset(train_path)
for x in train_ds.take(3):
    print(x)

for batch in builder.build_iter(train_ds).take(3):
    print(batch)

print(len(builder.text_feature.vocab))

text_embedder = tf.keras.layers.Embedding(len(builder.text_feature.vocab),
                                          200, mask_zero=True)

model = arcnlp.tf.models.BiLstmMatching(builder.features,
                                        builder.targets,
                                        text_embedder)
model.summary()
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=['acc'])
model.fit(builder.build_bucket_iter(train_ds), epochs=3)
