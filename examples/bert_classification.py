# -*- coding: utf-8 -*-

import os

import pandas as pd
import numpy as np
import tensorflow as tf
import arcnlp.tf

# tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

arcnlp.tf.utils.config_tf_gpu()


MAX_LEN = 128
BERT_MODEL_PATH = '/opt/userhome/ichongxiang/datasets/chinese_L-12_H-768_A-12'


def load_data(path):
    df = pd.read_csv(path, delimiter='\t')
    data_list = df.apply(
        lambda x: [''.join(x['text_a'].split()), int(x['label'])],
        axis=1).tolist()
    return data_list


train_data = load_data('/opt/userhome/ichongxiang/datasets/senta_data/train.tsv')
test_data = load_data('/opt/userhome/ichongxiang/datasets/senta_data/test.tsv')


tokenizer = arcnlp.tf.data.BertTokenizer(os.path.join(BERT_MODEL_PATH, 'vocab.txt'))


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
                text = d[0][:MAX_LEN]
                token_ids, type_ids = tokenizer.encode(first=text, max_len=MAX_LEN)
                token_ids_list.append(token_ids)
                type_ids_list.append(type_ids)
                label_list.append([d[1]])
                if len(label_list) == self.batch_size or i == idx_list[-1]:
                    token_ids_list = seq_padding(token_ids_list)
                    type_ids_list = seq_padding(type_ids_list)
                    label_list = seq_padding(label_list)
                    yield [token_ids_list, type_ids_list], label_list
                    token_ids_list, type_ids_list, label_list = [], [], []


train_gen = DataGenerator(train_data)
test_gen = DataGenerator(test_data)


def train_model():
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
    model.fit(
        train_gen.__iter__(),
        steps_per_epoch=len(train_gen),
        epochs=5,
        validation_data=test_gen.__iter__(),
        validation_steps=len(test_gen))
    model.save("./bert_model.h5")


def eval_model():
    custom_objects = {
        'BertEncoder': arcnlp.tf.layers.BertEncoder
    }
    model = tf.keras.models.load_model('./bert_model.h5',
                                       custom_objects=custom_objects)
    score = model.evaluate(test_gen.__iter__(),
                           steps=len(test_gen))
    print(dict(zip(model.metrics_names, score)))


if __name__ == '__main__':
    eval_model()
