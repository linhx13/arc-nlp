from typing import Iterable, Dict, Union
from itertools import chain

import tensorflow as tf

from ..fields import Field


class DataHandler:

    def __init__(self,
                 features: Dict[str, Field],
                 targets: Dict[str, Field]):
        self.features = features
        self.targets = targets

    def read_examples(self, path) -> Iterable[Dict]:
        raise NotImplementedError

    def build_vocab(self, *args, **kwargs):
        raise NotImplementedError

    def encode_example(self, example: Dict) -> Dict:
        raise NotImplementedError

    def build_dataset(self, data_source: Union[str, Iterable[Dict]]) -> tf.data.Dataset:
        if isinstance(data_source, str):
            examples = self.read_examples(data_source)
        else:
            examples = data_source
        examples = list(self.encode_example(ex) for ex in examples)

        def _gen():
            for ex in examples:
                yield ex

        return tf.data.Dataset.from_generator(
            _gen, output_types=self._output_types())

    def get_data_iter(self, dataset: tf.data.Dataset,
                      batch_size=32, train=True) -> tf.data.Dataset:
        padded_shapes = self._padded_shapes()
        padding_values = self._padding_values()
        if train:
            dataset = dataset.shuffle(batch_size * 100)
        dataset = dataset.padded_batch(batch_size,
                                       padded_shapes=padded_shapes,
                                       padding_values=padding_values)
        dataset = dataset.map(self._collate_fn,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def get_bucket_iter(self, dataset: tf.data.Dataset,
                        batch_size=32, train=True) -> tf.data.Dataset:
        padded_shapes = self._padded_shapes()
        padding_values = self._padding_values()
        if train:
            bucket_boundaries = self._bucket_boundaries(batch_size)
            bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)
            dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
                self.element_length_func, bucket_boundaries, bucket_batch_sizes,
                padded_shapes=padded_shapes, padding_values=padding_values))
            dataset = dataset.shuffle(100)
        else:
            dataset = dataset.padded_batch(
                batch_size, padded_shapes=padded_shapes)

        dataset = dataset.map(self._collate_fn,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def element_length_func(self, example) -> int:
        raise NotImplementedError

    def _bucket_boundaries(self, max_length, min_length=8, length_bucket_step=1.1):
        boundaries = []
        x = min_length
        while x < max_length:
            boundaries.append(x)
            x = max(x + 1, int(x * length_bucket_step))
        return boundaries

    def _padded_shapes(self):
        padded_shapes = {}
        for name, field in chain(self.features.items(), self.targets.items()):
            padded_shapes[name] = field.padded_shape()
        return padded_shapes

    def _padding_values(self):
        padding_values = {}
        for name, field in chain(self.features.items(), self.targets.items()):
            padding_values[name] = field.padding_value()
        return padding_values

    def _output_types(self):
        output_types = {}
        for name, field in chain(self.features.items(), self.targets.items()):
            output_types[name] = field.output_type()
        return output_types

    def _collate_fn(self, batch):
        features = {name: batch[name] for name in self.features.keys()}
        targets = {name: batch[name] for name in self.targets.keys()}
        return features, targets
