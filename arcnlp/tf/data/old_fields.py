# -*- coding: utf-8 -*-

import six
from collections import Counter, OrderedDict
from itertools import chain

import numpy as np

from ..vocabs import Vocab
from .dataset import Dataset
from . import utils


class Field():
    """Defines a datatype together with instructions for converting to Tensor.

    Field class models common text processing datatypes that can be represented
    by tensors.  It holds a Vocab object that defines the set of possible values
    for elements of the field and their corresponding numerical representations.
    The Field object also holds other parameters relating to how a datatype
    should be numericalized, such as a tokenization method and the kind of
    Tensor that should be produced.

    If a Field is shared between two columns in a dataset (e.g., question and
    answer in a QA dataset), then they will have a shared vocabulary.
    """

    vocab_cls = Vocab

    def __init__(self, sequential=True, use_vocab=True, init_token=None,
                 eos_token=None, fix_length=None, dtype='int32',
                 preprocessing=None, postprocessing=None, lower=False,
                 tokenize=None, include_lengths=False, unk_token='<unk>',
                 pad_token='<pad>', pad_first=False, truncate_first=False,
                 is_target=False, namespace='tokens'):
        self.sequential = sequential
        self.use_vocab = use_vocab
        self.init_token = init_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.fix_length = fix_length
        self.dtype = dtype
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.lower = lower
        self.tokenize = tokenize if tokenize else utils.split_tokenizer
        self.include_lengths = include_lengths
        self.pad_token = pad_token if self.sequential else None
        self.pad_first = pad_first
        self.truncate_first = truncate_first
        self.is_target = is_target
        self.namespace = namespace

    def __hash__(self):
        return 42

    def __eq__(self, other):
        if not isinstance(other, Field):
            return False
        return self.__dict__ == other.__dict__

    def preprocess(self, x):
        if (six.PY2 and isinstance(x, six.string_types)
                and not isinstance(x, six.text_type)):
            if isinstance(x, list):
                x = [six.text_type(s, encoding='utf-8') for s in x]
            else:
                x = six.text_type(x, encoding='utf-8')
        if self.sequential and isinstance(x, six.text_type):
            x = self.tokenize(x.rstrip('\n'))
        if self.lower:
            if isinstance(x, list):
                x = [s.lower() for s in x]
            else:
                x = x.lower()
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch):
        """Process a list of examples to create a np.ndarray."""
        padded = self.pad(batch)
        tensor = self.numericalize(padded)
        return tensor

    def pad(self, minibatch):
        """Pad a batch of examples using this field."""
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
                    [self.pad_token] * max(0, max_len - len(x))
                    + ([] if self.init_token is None else [self.init_token])
                    + list(x[-max_len:] if self.truncate_first else x[:max_len])
                    + ([] if self.eos_token is None else [self.eos_token]))
            else:
                padded.append(
                    ([] if self.init_token is None else [self.init_token])
                    + list(x[-max_len:] if self.truncate_first else x[:max_len])
                    + ([] if self.eos_token is None else [self.eos_token])
                    + [self.pad_token] * max(0, max_len - len(x)))
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        if self.include_lengths:
            return (padded, lengths)
        return padded

    def numericalize(self, arr):
        """Turn a batch of examples that use this field into np.ndarray.
        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.
        """
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths seet to True, but"
                             "input data is not a tuple of "
                             "(data_batch, batch_lengths).")
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
                numericalization_func = getattr(np, self.dtype)
                arr = [numericalization_func(x) if isinstance(x, six.string_types)
                       else x for x in arr]
            if self.postprocessing is not None:
                arr = self.postprocessing(arr, None)

        var = np.array(arr, dtype=self.dtype)

        if self.include_lengths:
            return var, lengths
        return var

    def build_vocab(self, *args, **kwargs):
        """Construct the Vocab object for this field from one or more datasets.
        """
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
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

     A label field is a shallow wrapper around a standard field designed to hold labels
    for a classification task. Its only use is to set the unk_token and sequential to
    `None` by default.
    """

    def __init__(self, namespace="labels", **kwargs):
        kwargs['sequential'] = False
        kwargs['unk_token'] = None
        kwargs['is_target'] = True
        super(LabelField, self).__init__(**kwargs)
        self.namespace = namespace
