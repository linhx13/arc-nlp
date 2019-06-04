# -*- coding: utf-8 -*-

from collections import Counter, OrderedDict
from itertools import chain

import numpy as np
import six

from .. import constants as C
from ..vocab import Vocab
from .dataset import Dataset


class RawField(object):
    """Defines a general datatype.

    Every dataset consists of one or more types of data. For instance, a text
    classification dataset contains sentences and their classes, while a
    machine translation dataset contains paired examples of text in two
    languages. Each of these types of data is represented by a RawField object.
    A RawField object does not assume any property of the data type and
    it holds parameters relating to how a datatype should be processed.

    Attributes:
        preprocessing: The Pipeline that will be applied to examples
            using this field before create an example.
            Default: None.
        postprocessing: A Pipeline that will be applied to a list of examples
            using this field before assigning to a batch.
            Function signature: (batch(list)) -> object. Default: None.
        is_target: Whether this field is a target variable.
            Affects iteration over batches. Default: False.
    """

    def __init__(self, preprocessing=None, postprocessing=None, is_target=False):
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.is_target = is_target

    def preprocess(self, x):
        """Preprocess an example if the `preprocessing` is provided."""
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, *args, **kwargs):
        """Process a list of examples to create a batch.

        Postprocess the batch with user-provided Pipeline.

        Args:
            batch (list(object)): A list of object from a batch of examples.
        Returns:
            object: Processed object given the input and custom
                postprocessing Pipeline.
        """
        if self.postprocessing is None:
            batch = self.postprocessing(batch)
        return batch


class Field(RawField):

    vocab_cls = Vocab

    def __init__(self, sequential=True, use_vocab=True, init_token=None,
                 eos_token=None, fix_length=None, dtype=np.int32,
                 preprocessing=None, postprocessing=None, lower=False,
                 tokenize=None, include_lengths=False,
                 pad_token=C.PAD_TOKEN, unk_token=C.UNK_TOKEN,
                 pad_first=False, truncate_first=False, stop_words=None,
                 is_target=False):
        self.sequential = sequential
        self.use_vocab = use_vocab
        self.init_token = init_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token if self.sequential else None
        self.fix_length = fix_length
        self.dtype = dtype
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.lower = lower
        self.tokenize = tokenize \
            if tokenize is not None else six.text_type.split
        self.include_lengths = include_lengths
        self.pad_first = pad_first
        self.truncate_first = truncate_first
        try:
            self.stop_words = set(stop_words) \
                              if stop_words is not None else None
        except TypeError:
            raise ValueError("Stop words must be convertible to a set")
        self.is_target = is_target

    def __eq__(self, other):
        if not isinstance(other, Field):
            return False

        return self.__dict__ == other.__dict__

    def preprocess(self, x):
        """Load a single example using this field, tokenizing if necessary."""
        if six.PY2 and isinstance(x, six.string_types) \
           and not isinstance(x, six.text_type):
            x = six.text_type(x, encoding='utf-8')
        if self.sequential and isinstance(x, six.string_types):
            x = self.tokenize(x.rstrip('\r\n'))
        if self.lower:
            x = six.text_type.lower(x)
        if self.sequential and self.use_vocab and self.stop_words is not None:
            x = [w for w in x if w not in self.stop_words]
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch):
        """Process a list of examples to create a numpy.ndarray.

        Pad, numericalize, and postprocesses a batch and create a ndarray.

        Args:
            batch (list(example)): A list of object from a batch of examples.

        Returns:
            A numpy.ndarray from the batch.
        """
        padded = self.pad(batch)
        tensor = self.numericalize(padded)
        return tensor

    def pad(self, minibatch):
        """Pad a batch of examples using this field.

        Pads to self.fix_length if provided, otherwise pads to the length of
        the longest example in the batch. Prepends self.init_token and appends
        self.eos_token if those attributes are not None. Returns a tuple of the
        padded list and a list containing lengths of each example if
        `self.include_lengths` is `True` and `self.sequential` is `True`, else
        just returns the padded list. If `self.sequential` is `False`, no
        padding is applied.
        """
        minibatch = list(minibatch)
        if not self.sequential:
            return minibatch
        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = self.fix_length + \
                      (self.init_token, self.eos_token).count(None) - 2
        padded, lengths = [], []
        for x in minibatch:
            if self.pad_first:
                padded.append(
                    [self.pad_token] * max(0, max_len - len(x)),
                    + ([] if self.init_token is None else [self.init_token])
                    + (x[-max_len:] if self.truncate_first else x[:max_len])
                    + ([] if self.eos_token is None else [self.eos_token]))
            else:
                padded.append(
                    ([] if self.init_token is None else [self.init_token])
                    + (x[-max_len:] if self.truncate_first else x[:max_len])
                    + ([] if self.eos_token is None else [self.eos_token])
                    + [self.pad_token] * max(0, max_len - len(x)))
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        if self.include_lengths:
            return (padded, lengths)
        else:
            return padded

    def numericalize(self, arr):
        """Turn a batch of examples that use this field into a numpy.ndarray.

        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.
        """
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch length).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = np.array(lengths, dtype=self.dtype)

        if self.use_vocab:
            if self.sequential:
                arr = [[self.vocab.stoi[x] for x in ex] for ex in arr]
            else:
                arr = [self.vocab.stoi[x] for x in arr]
            if self.postprocessing is not None:
                arr = self.postprocessing(arr, self.vocab)
        else:
            if not self.sequential:
                arr = [self.dtype(x) if isinstance(x, six.string_types)
                       else x for x in arr]
            if self.postprocessing is not None:
                arr = self.postprocessing(arr, None)

        arr = np.array(arr, dtype=self.dtype)
        if self.include_lengths:
            return arr, lengths
        else:
            return arr

    def build_vocab(self, *args, **kwargs):
        """Construct the Vocab object for this field from one or more datasets.

        Args:
            args: Dataset objects or other iterable data sources from which to
                construct the Vocab object that represents the set of possible
                values for this field. If a Dataset object is provided, all
                columns corresponding to this field are used; individual column
                can also be provided directly.
            kwargs: Passed to the constructor of Vocab.
        """
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name)
                            for name, field in arg.fields.items()
                            if field is self]
            else:
                sources.append(arg)
        for data in sources:
            for x in data:
                if not self.sequential:
                    x = [x]
                try:
                    counter.update(x)
                except TypeError:
                    counter.update(chain.from_iterable(x))
        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.pad_token, self.unk_token, self.init_token,
                            self.eos_token] + kwargs.pop('specials', [])
            if tok is not None))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)



class LabelField(Field):
    """A Label field.

    A label field is a shallow wrapper around a standard field designed to hold
    labels for a classification task. Its only use to set the unk_token and
    sequential to `None` by default.
    """

    def __init__(self, **kwargs):
        kwargs['sequential'] = False
        kwargs['unk_token'] = None
        kwargs['is_target'] = True
        super(LabelField, self).__init__(**kwargs)
