# -*- coding: utf-8 -*-

from tensorflow.keras.utils import Sequence

from . import Dataset, DataHandler, BucketIterator, Batch


class DataGenerator(object):

    def __init__(self, data_handler: DataHandler):
        self.data_handler = data_handler

    def create_data_arrays(self, dataset: Dataset, batch_size: int,
                           train: bool = True):
        return _get_batch_fields(Batch(dataset.examples, dataset),
                                 self.data_handler)

    def create_data_generator(self, dataset: Dataset, batch_size,
                              train: bool = True):
        if train:
            kwargs = {'train': True, 'repeat': True,
                      'sort_within_batch': True}
        else:
            kwargs = {'train': False, 'repeat': True, 'sort': False}
        data_iter = BucketIterator(dataset, batch_size=batch_size,
                                   sort_key=self.data_handler.sort_key,
                                   **kwargs)
        return self._data_generator(data_iter), data_iter

    def create_data_sequence(self, dataset: Dataset, batch_size,
                             train: bool = True):
        return DataSequence(dataset, self.data_handler, batch_size, train)

    def _data_generator(self, data_iter):
        for batch in data_iter:
            yield _get_batch_fields(batch, self.data_handler)


class DataSequence(Sequence):

    def __init__(self, dataset: Dataset,
                 data_handler: DataHandler,
                 batch_size: int,
                 train: bool = True):
        self.dataset = dataset
        self.data_handler = data_handler
        self.batch_size = batch_size
        if train:
            kwargs = {'train': True, 'repeat': False}
        else:
            kwargs = {'train': False, 'repeat': False, 'sort': False}
        self.data_iter = BucketIterator(dataset, batch_size=batch_size,
                                        sort_key=data_handler.sort_key,
                                        **kwargs)
        self.on_epoch_end()

    def __len__(self):
        return len(self.data_iter)

    def on_epoch_end(self):
        self.data_iter.create_batches()
        self.batches = list(self.data_iter.batches)

    def __getitem__(self, idx):
        minibatch = Batch(self.batches[idx], self.dataset)
        return _get_batch_fields(minibatch, self.data_handler)


def _get_batch_fields(batch, data_handler):
    features = dict((f, getattr(batch, f))
                    for f in data_handler.features)
    targets = dict((f, getattr(batch, f))
                   for f in data_handler.targets)
    return features, targets
