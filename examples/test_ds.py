import os
from typing import Dict

import tensorflow as tf


train_path = os.path.expanduser("~/datasets/LCQMC/train.txt")
print(train_path)


class Feature:

    def __init__(self):
        pass


def gen(data_iter, max_cnt=10):
    cnt = 0
    for idx, data in enumerate(data_iter):
        yield data
        cnt += 1
        if cnt >= max_cnt:
            break


class LCQMC:

    def __init__(self, features, targets):
        pass

    def build_dataset(self, path) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_generator(
            lambda: self.read_examples(path),
            output_types=self.example_output_types(),
        )
        return ds

    def read_examples(self, path) -> Dict:
        with open(path) as fin:
            for line in fin:
                line = line.strip('\r\n')
                if not line:
                    continue
                arr = line.split('\t')
                yield self.build_example(arr[0], arr[1], arr[2])

    def example_output_types(self):
        return {"premise": tf.string, 'hypothesis': tf.string, 'label': tf.int32}

    def build_example(self, premise, hypothesis, label=None) -> Dict:
        ex = {}
        ex['premise'] = premise
        ex['hypothesis'] = hypothesis
        if label is not None:
            ex['label'] = int(label)
        return ex

    def build_vocab(self):
        pass


builder = LCQMC(None, None)
ds = builder.build_dataset(train_path)

for idx, ex in enumerate(ds.repeat().take(20)):
    print('=' * 10)
    print('idx: %d' % idx)
    print(ex)
