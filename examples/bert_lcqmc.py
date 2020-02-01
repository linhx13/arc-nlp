# -*- coding: utf-8 -*-

import os

from tqdm import tqdm
import tensorflow as tf
import arcnlp.tf
import numpy as np


tf.compat.v1.disable_eager_execution()

arcnlp.tf.utils.config_tf_gpu()


MAX_LEN = 128
BERT_MODEL_PATH = '/opt/userhome/ichongxiang/datasets/chinese_L-12_H-768_A-12'


def load_data(path):
    data_list = []
    with open(path) as fin:
        for line in tqdm(fin):
            left, right, label = line.split('\t')
            data_list.append([left, right, int(label)])
    return data_list


tokenizer = arcnlp.tf.data.BertTokenizer(
    os.path.join(BERT_MODEL_PATH, 'vocab.txt'))


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class DataGenerator():
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idx_list = list(range(len(self.data)))
            np.random.shuffle(idx_list)
            token_ids_list, type_ids_list, label_list = [], [], []
            for i in idx_list:
                d = self.data[i]
                # text = d[0][:MAX_LEN]
                left, right, label = d
                token_ids, type_ids = tokenizer.encode(first=left, second=right,
                                                       max_len=MAX_LEN)
                token_ids_list.append(token_ids)
                type_ids_list.append(type_ids)
                label_list.append([label])
                if len(label_list) == self.batch_size or i == idx_list[-1]:
                    token_ids_list = seq_padding(token_ids_list)
                    type_ids_list = seq_padding(type_ids_list)
                    label_list = seq_padding(label_list)
                    yield [token_ids_list, type_ids_list], label_list
                    token_ids_list, type_ids_list, label_list = [], [], []


def train_model():
    train_gen = DataGenerator(
        load_data('/opt/userhome/ichongxiang/datasets/LCQMC/train.txt'))
    test_gen = DataGenerator(
        load_data('/opt/userhome/ichongxiang/datasets/LCQMC/dev.txt'))
    token_ids_input = tf.keras.layers.Input(shape=(None,))
    type_ids_input = tf.keras.layers.Input(shape=(None,))
    encoder = arcnlp.tf.layers.BertEncoder(BERT_MODEL_PATH, seq_len=MAX_LEN)
    x = encoder([token_ids_input, type_ids_input])
    preds = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=[token_ids_input, type_ids_input],
                                  outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(1e-5),
                  metrics=['accuracy'])
    model.summary()
    model.fit(train_gen.__iter__(),
              steps_per_epoch=len(train_gen),
              epochs=5,
              validation_data=test_gen.__iter__(),
              validation_steps=len(test_gen))
    model.save('./bert_lcqmc_model.h5')


if __name__ == '__main__':
    train_model()
