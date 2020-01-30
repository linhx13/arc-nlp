# -*- coding: utf-8 -*-

from collections import Counter, OrderedDict
from itertools import chain

import torchtext
from .dataset import Dataset


class Field(torchtext.data.Field):

    def __init__(self, namespace="tokens", **kwargs):
        kwargs['batch_first'] = True
        super(Field, self).__init__(**kwargs)
        self.namespace = namespace

    def numericalize(self, arr, device=None):
        var = super(Field, self).numericalize(arr, device)
        if self.include_lengths:
            return var[0].numpy(), var[1]
        else:
            return var.numpy()

    def build_vocab(self, *args, **kwargs):
        """Construct the Vocab object for this field from one or more datasets.

        Arguments:
            Positional arguments: Dataset objects or other iterable data
                sources from which to construct the Vocab object that
                represents the set of possible values for this field. If
                a Dataset object is provided, all columns corresponding
                to this field are used; individual columns can also be
                provided directly.
            Remaining keyword arguments: Passed to the constructor of Vocab.
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

    def __init__(self, namespace="labels", **kwargs):
        kwargs['batch_first'] = True
        kwargs['sequential'] = False
        kwargs['unk_token'] = None
        kwargs['is_target'] = True
        super(LabelField, self).__init__(**kwargs)
        self.namespace = namespace
