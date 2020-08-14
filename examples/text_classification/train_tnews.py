import json
import os
from collections import Counter

import tensorflow as tf
import jieba

from arcnlp.tf.datasets import TextClassificationDataset
from arcnlp.tf.data.functional import sequential_transforms, vocab_func, totensor, label_func
from arcnlp.tf.vocab import Vocab, build_vocab_from_iterator
from arcnlp.tf.layers import BOWEncoder, CNNEncoder

jieba.initialize()

label_path = os.path.expanduser("~/datasets/tnews_public/labels.json")
train_path = os.path.expanduser("~/datasets/tnews_public/train.json")
val_path = os.path.expanduser("~/datasets/tnews_public/dev.json")


def tokenizer(text):
    text = tf.compat.as_text(text)
    return jieba.lcut(text)


def build_dataset(data_path, label_path, text_vocab=None, label_vocab=None):
    examples = []
    with open(data_path) as fin:
        for line in fin:
            data = json.loads(line)
            example = ('%s:%s' % (data['label'], data['label_desc']), data['sentence'])
            examples.append(example)

    text_transform = sequential_transforms(tokenizer)
    if text_vocab is None:
        text_vocab = build_vocab_from_iterator(text_transform(ex[1]) for ex in examples)

    if label_vocab is None:
        label_counter = Counter()
        with open(label_path) as fin:
            for line in fin:
                data = json.loads(line)
                label_counter['%s:%s' % (data['label'], data['label_desc'])] += 1
        label_vocab = Vocab(label_counter, specials=[])
    print(label_vocab.stoi)

    label_transform = sequential_transforms(
        label_func(label_vocab), totensor(dtype='int32'))
    text_transform = sequential_transforms(
        text_transform, vocab_func(text_vocab), totensor(dtype='int32'))
    dataset = TextClassificationDataset(examples, (label_vocab, text_vocab),
                                        (label_transform, text_transform))
    return dataset, text_vocab, label_vocab


train_dataset, text_vocab, label_vocab = build_dataset(train_path, label_path)
val_dataset, _, _ = build_dataset(val_path, label_path, text_vocab=text_vocab, label_vocab=label_vocab)


def Model(label_vocab, text_vocab, dropout=0.5):
    input_tokens = tf.keras.layers.Input(shape=(None,), name='tokens')
    embed_tokens = tf.keras.layers.Embedding(len(text_vocab), 200, mask_zero=True)(input_tokens)
    encoded_tokens = CNNEncoder()(embed_tokens)
    if dropout:
        encoded_tokens = tf.keras.layers.Dropout(dropout)(encoded_tokens)
    probs = tf.keras.layers.Dense(len(label_vocab), activation='softmax', name='label')(encoded_tokens)
    return tf.keras.models.Model(inputs=[input_tokens], outputs=[probs])


model = Model(label_vocab, text_vocab)
model.summary()


def get_iter(dataset, label_vocab, text_vocab, batch_size=32, train=True):
    output_types= (tf.int32, tf.int32)
    padded_shapes = ([None], [None])
    padding_values = (0, text_vocab['<pad>'])

    raw_dataset = dataset

    def _gen():
        for idx in range(len(raw_dataset)):
            yield raw_dataset[idx][1], raw_dataset[idx][0]

    dataset = tf.data.Dataset.from_generator(_gen, output_types=output_types)

    if train:
        dataset = dataset.shuffle(batch_size * 100)
    dataset = dataset.padded_batch(batch_size,
                                   padded_shapes=padded_shapes,
                                   padding_values=padding_values)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


train_iter = get_iter(train_dataset, label_vocab, text_vocab, train=True)
val_iter = get_iter(val_dataset, label_vocab, text_vocab, train=False)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.fit(train_iter,
          validation_data=val_iter,
          epochs=3)
