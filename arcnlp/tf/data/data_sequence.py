# -*- coding: utf-8 -*-

from tensorflow.keras.utils import Sequence

from . import Dataset, BucketIterator, Batch


class DataSequence(Sequence):

    def __init__(self, dataset: Dataset,
                 data_handler,
                 batch_size: int,
                 train: bool = True):
        self.dataset = dataset
        self.data_handler = data_handler
        self.batch_size = batch_size
        if train:
            kwargs = {'train': True, 'repeat': False,
                      'sort_within_batch': True}
        else:
            kwargs = {'train': False, 'repeat': False, 'sort': False, }
        self.data_iter = BucketIterator(dataset, batch_size=batch_size,
                                        sort_key=data_handler.sort_key,
                                        **kwargs)
        self.on_epoch_end()

    def __len__(self):
        return len(self.data_iter)

    def __getitem__(self, idx):
        minibatch = Batch(self.batches[idx], self.dataset)
        return _get_batch_fields(minibatch, self.data_handler)

    def on_epoch_end(self):
        self.data_iter.create_batches()
        self.batches = list(self.data_iter.batches)


def _get_batch_fields(batch, data_handler):
    features = dict((f, getattr(batch, f))
                    for f in data_handler.features)
    targets = dict((f, getattr(batch, f))
                   for f in data_handler.targets)
    return features, targets
