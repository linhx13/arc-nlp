from typing import Iterable, Dict
from itertools import chain

import tensorflow as tf

from ..transforms import Transform


class DatasetBuilder:

    def __init__(self,
                 features: Dict[str, Transform],
                 targets: Dict[str, Transform]):
        self.features = features
        self.targets = targets

    def read_from_path(self, path) -> Iterable[Dict]:
        raise NotImplementedError

    def build_example(self, **inputs) -> Dict:
        return inputs

    def build_vocab(self, *examples):
        raise NotImplementedError

    def encode_example(self, example: Dict) -> Dict:
        res = {}
        for name, transform in chain(self.features.items(), self.targets.items()):
            res[name] = transform.encode(example[name])
        return res

    def build_dataset(self, path) -> tf.data.Dataset:
        examples = list(self.read_from_path(path))

        def _gen():
            for ex in examples:
                yield self.encode_example(ex)

        output_types = self.output_types()
        return tf.data.Dataset.from_generator(_gen, output_types=output_types)

    def build_iter(self, dataset, batch_size=32, train=True) -> tf.data.Dataset:
        padded_shapes = self.padded_shapes()
        if train:
            dataset = dataset.shuffle(batch_size * 100)
        dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
        dataset = dataset.map(self._batch_pair_fn)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def build_bucket_iter(self, dataset, batch_size=32, train=True) -> tf.data.Dataset:
        padded_shapes = self.padded_shapes()
        if train:
            bucket_boundaries = self._bucket_boundaries(batch_size)
            bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)
            dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
                self.element_length_func, bucket_boundaries, bucket_batch_sizes,
                padded_shapes=padded_shapes))
            dataset = dataset.shuffle(10)
        else:
            dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)

        dataset = dataset.map(self._batch_pair_fn)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def _bucket_boundaries(self, max_length, min_length=8, length_bucket_step=1.1):
        boundaries = []
        x = min_length
        while x < max_length:
            boundaries.append(x)
            x = max(x + 1, int(x * length_bucket_step))
        return boundaries

    def element_length_func(self, example) -> int:
        raise NotImplementedError

    def padded_shapes(self):
        padded_shapes = {}
        for name, transform in self.features.items():
            padded_shapes[name] = transform.padded_shape()
        for name, transform in self.targets.items():
            padded_shapes[name] = transform.padded_shape()
        return padded_shapes

    def output_types(self):
        output_types = {}
        for name, transform in self.features.items():
            output_types[name] = transform.output_type()
        for name, transform in self.targets.items():
            output_types[name] = transform.output_type()
        return output_types

    def _batch_pair_fn(self, batch):
        features = {name: transform.postprocessing(batch[name])
                    for name, transform in self.features.items()}
        targets = {name: transform.postprocessing(batch[name])
                   for name, transform in self.targets.items()}
        return features, targets
