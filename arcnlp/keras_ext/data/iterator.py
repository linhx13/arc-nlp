# -*- coding: utf-8 -*-


class Iterator(object):
    """Define an iterator that loads batches of data from Dataset.

    Attributes:
        dataset: The Dataset object to load Examples from.
        batch_size: Batch size.
    """
