# -*- coding: utf-8 -*-

import random
from contextlib import contextmanager
from copy import deepcopy
import collections


class RandomShuffler(object):
    """Use random functions while keeping track of the random state to make it
    reproducible and deterministic."""

    def __init__(self, random_state=None):
        self._random_state = random_state
        if self._random_state is None:
            self._random_state = random.getstate()

    @contextmanager
    def use_internal_state(self):
        """Use a specific RNG state."""
        old_state = random.getstate()
        random.setstate(self._random_state)
        yield
        self._random_state = random.getstate()
        random.setstate(old_state)

    @property
    def random_state(self):
        return deepcopy(self._random_state)

    @random_state.setter
    def random_state(self, s):
        self._random_state = s

    def __call__(self, data):
        """Shuffle and return a new list."""
        with self.use_internal_state():
            return random.sample(data, len(data))


def split_tokenizer(x):
    return x.split()


class DefaultLookupDict(dict):
    """Dictionary class with fall-back lookup with default value set in the constructor."""

    def __init__(self, default, d=None):
        if d:
            super(DefaultLookupDict, self).__init__(d)
        else:
            super(DefaultLookupDict, self).__init__()
        self._default = default

    def __getitem__(self, k):
        return self.get(k, self._default)


class Counter(collections.Counter):

    def discard(self, min_freq, unknown_token):
        freq = 0
        ret = Counter({})
        for token, count in self.items():
            if count < min_freq:
                freq += count
            else:
                ret[token] = count
        ret[unknown_token] = ret.get(unknown_token, 0) + freq
        return ret


def count_tokens(tokens, to_lower=False, counter=None):
    if to_lower:
        tokens = [t.lower() for t in tokens]

    if counter is None:
        return Counter(tokens)
    else:
        counter.update(tokens)
        return counter
