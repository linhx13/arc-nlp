import json
import os
from collections import Counter
import math

import tensorflow as tf
import numpy as np
import jieba

from arcnlp.tf.datasets import TextClassificationDataset
from arcnlp.tf.data.functional import sequential_transforms, vocab_func, totensor, label_func
from arcnlp.tf.vocab import Vocab, build_vocab_from_iterator
from arcnlp.tf.layers import BOWEncoder #, CNNEncoder

jieba.initialize()

tf.compat.v1.disable_eager_execution()

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
        label_func(label_vocab)) #, totensor(dtype='int32'))
    text_transform = sequential_transforms(
        text_transform, vocab_func(text_vocab)) # , totensor(dtype='int32'))
    dataset = TextClassificationDataset(examples, (label_vocab, text_vocab),
                                        (label_transform, text_transform))
    return dataset, text_vocab, label_vocab


train_dataset, text_vocab, label_vocab = build_dataset(train_path, label_path)
val_dataset, _, _ = build_dataset(val_path, label_path, text_vocab=text_vocab, label_vocab=label_vocab)
print(len(train_dataset))
print(len(val_dataset))


class CNNEncoder(tf.keras.layers.Layer):

    def __init__(self, filters=100, kernel_sizes=(2, 3, 4, 5),
                 conv_layer_activation='relu',
                 l1_regularization=None, l2_regularization=None,
                 **kwargs):
        super(CNNEncoder, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.conv_layer_activation = conv_layer_activation
        self.l1_regularization = l1_regularization
        self.l2_regularization = l2_regularization

        self.regularizer = tf.keras.regularizers.l1_l2(
            l1=l1_regularization if l1_regularization is not None else 0.0,
            l2=l2_regularization if l2_regularization is not None else 0.0)
        self.conv_layers = [tf.keras.layers.Conv1D(filters=self.filters,
                                                   kernel_size=kernel_size,
                                                   activation=self.conv_layer_activation,
                                                   kernel_regularizer=self.regularizer,
                                                   bias_regularizer=self.regularizer)
                            for kernel_size in self.kernel_sizes]

    def call(self, inputs, mask=None):
        conv_outputs = [tf.keras.backend.max(conv_layer(inputs), axis=1)
                        for conv_layer in self.conv_layers]
        maxpool_output = tf.keras.layers.concatenate(conv_outputs) \
            if len(conv_outputs) > 1 else conv_outputs[0]
        return maxpool_output

    def get_config(self):
        config = {"filters": self.filters,
                  "kernel_sizes": self.kernel_sizes,
                  "conv_layer_activation": self.conv_layer_activation,
                  "l1_regularization": self.l1_regularization,
                  "l2_regularization": self.l2_regularization
                  }
        base_config = super(CNNEncoder, self).get_config()
        config.update(base_config)
        return config


def FunctionalModel(label_vocab, text_vocab, dropout=0.5):
    input_tokens = tf.keras.layers.Input(shape=(None,), name='tokens')
    embed_tokens = tf.keras.layers.Embedding(len(text_vocab), 200, mask_zero=True)(input_tokens)
    encoded_tokens = CNNEncoder()(embed_tokens)
    if dropout:
        encoded_tokens = tf.keras.layers.Dropout(dropout)(encoded_tokens)
    probs = tf.keras.layers.Dense(len(label_vocab), activation='softmax', name='label')(encoded_tokens)
    return tf.keras.models.Model(inputs=[input_tokens], outputs=[probs])


class Model(tf.keras.Model):

    def __init__(self, label_vocab, text_vocab, dropout=0.5):
        super(Model, self).__init__()
        self.embedding = tf.keras.layers.Embedding(len(text_vocab), 200, mask_zero=True)
        self.encoder = CNNEncoder()
        self.dropout = tf.keras.layers.Dropout(dropout) if dropout else None
        self.dense = tf.keras.layers.Dense(len(label_vocab), activation='softmax', name='label')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.encoder(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.dense(x)


model = Model(label_vocab, text_vocab)
# model.summary()


def get_iter(dataset, label_vocab, text_vocab, batch_size=32, train=True):
    output_types= (tf.int32, tf.int32)
    padded_shapes = ([None], [None])
    padding_values = (0, text_vocab['<pad>'])

    examples = [dataset[idx][::-1] for idx in range(len(dataset))]

    def _gen():
        for example in examples:
            yield example

    dataset = tf.data.Dataset.from_generator(_gen, output_types=output_types)

    if train:
        dataset = dataset.shuffle(batch_size * 100)
    dataset = dataset.padded_batch(batch_size,
                                   padded_shapes=padded_shapes,
                                   padding_values=padding_values)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset, math.ceil(len(examples) / batch_size)


train_iter, train_steps = get_iter(train_dataset, label_vocab, text_vocab, train=True)
val_iter, val_steps = get_iter(val_dataset, label_vocab, text_vocab, train=False)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

val_metric = 'val_acc'
early_stop = tf.keras.callbacks.EarlyStopping(monitor=val_metric,
                                              patience=3)
checkpoint_path = './test_model/checkpoints/ckpt-{epoch:04d}.ckpt'
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                      save_best_only=True,
                                                      save_weights_only=False,
                                                      monitor=val_metric)
callbacks = [early_stop, model_checkpoint]

history = model.fit(train_iter,
                    validation_data=val_iter,
                    epochs=10,
                    # steps_per_epoch=train_steps,
                    validation_steps=val_steps,
                    callbacks=callbacks)

best_epoch = int(np.argmax(history.history[val_metric])) + 1
print('best_epoch: {:d}'.format(best_epoch))
model.load_weights(checkpoint_path.format(epoch=best_epoch))

model.save("./test_model/model")
new_model = tf.keras.models.load_model('./test_model/model')
print(new_model.evaluate(val_iter))
