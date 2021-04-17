import os
import math

import tensorflow as tf

from arcnlp.tf.datasets import SequenceTaggingDataset
from arcnlp.tf.datasets.sequence_tagging import create_data_from_iob, build_vocab
from arcnlp.tf.vocab import Vocab, build_vocab_from_iterator
from arcnlp.tf.data.functional import sequential_transforms, label_func, vocab_func

train_path = os.path.expanduser("~/datasets/conll2000/train.txt")
val_path = os.path.expanduser("~/datasets/conll2000/test.txt")


def build_dataset(data_path, vocabs=None, transforms=None):
    examples = list(create_data_from_iob(data_path, " "))

    if vocabs is None:
        examples = list(examples)
        vocabs = []
        for idx in range(len(examples[0])):
            vocab = build_vocab_from_iterator(ex[idx] for ex in examples)
            vocabs.append(vocab)
    print(vocabs[-1].stoi)

    if transforms is None:
        transforms = []
        for idx in range(len(examples[0])):
            if idx == len(examples[0]) - 1:
                transform = label_func(vocabs[idx], sparse=True)
            else:
                transform = vocab_func(vocabs[idx])
            transforms.append(transform)

    dataset = SequenceTaggingDataset(examples, vocabs, transforms)
    return dataset, vocabs, transforms


train_dataset, vocabs, transforms = build_dataset(train_path)
val_dataset, _, _ = build_dataset(val_path, vocabs, transforms)
print(len(train_dataset))
print(len(val_dataset))


def collate_fn(self, batch):
    features = batch[0:-1]
    targets = batch[-1]
    return features, targets


def get_iter(dataset, vocabs, batch_size=32, train=True):
    examples = [tuple(dataset[idx]) for idx in range(len(dataset))]
    for idx in range(5):
        print(examples[idx])

    for ex in examples:
        assert len(ex[0]) == len(ex[1]) and len(ex[1]) == len(ex[2])

    output_types = (tf.int32, tf.int32, tf.int32)

    def _gen():
        for example in examples:
            yield example

    dataset = tf.data.Dataset.from_generator(_gen, output_types=output_types)
    # for data in dataset.take(5):
    #     print(data)

    for data in dataset:
        assert data[0].shape == data[1].shape and data[1].shape == data[2].shape

    padded_shapes = ([None], [None], [None])
    padding_values = (vocabs[0]['<pad>'], vocabs[1]['<pad>'], vocabs[2]['<pad>'])
    if train:
        dataset = dataset.shuffle(batch_size * 100)
    dataset = dataset.padded_batch(batch_size,
                                   padded_shapes=padded_shapes,
                                   padding_values=padding_values)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset, math.ceil(len(examples) / batch_size)


train_iter, train_steps = get_iter(train_dataset, vocabs, train=True)
val_iter, val_steps = get_iter(val_dataset, vocabs, train=False)

for idx, batch in enumerate(train_iter):
    print(batch)
    if idx == 5:
        break
