# -*- coding: utf-8 -*-

import logging
from collections import defaultdict

from . import constants as C

logger = logging.getLogger(__name__)


class Vocab(object):
    """A Vocab object will be used to index text tokens.

    Attributes:
        stoi: A collections.defaultdict instance mapping token string to
            numerical identifiers.
        itos: A list of token strings indexed by their numerical identifiers.
        unk_token: The representation for any unknown token.

    """

    UNK = C.UNK_TOKEN

    def __init__(self, counter=None, max_size=None, min_freq=1,
                 specials=[C.PAD_TOKEN]):
        """Create a Vocab object from a collections.Counter.

        Arguments:
            counter: collections.Counter object holding the frequencies of
                each value found in the data.
            max_size: The maximum size of the most frequent tokens of the
                vocabulary, or None for no maximum. Note this argument does not
                count any token from `specials`. Default: None.
            min_freq: The minimum frequency needed to include a token in the
                vocabulary. Values less than 1 will be set to 1. Default: 1.
            specials: The list of special tokens will be prepended to the
                vocabulary. Default: ['<pad>'].
        """
        min_freq = max(min_freq, 1)
        max_size = None if max_size is None else max_size + len(specials)

        self.itos = list(specials)
        specials_set = set(specials)
        word_freqs = sorted(counter.items(), key=lambda t: t[1], reverse=True)
        for word, freq in word_freqs:
            if freq < min_freq or len(self.itos) == max_size:
                break
            if word in specials_set:
                continue
            self.itos.append(word)

        self.unk_index, self.unk_token = None, None
        if Vocab.UNK in specials:
            self.unk_index = specials.index(Vocab.UNK)
            self.unk_token = Vocab.UNK
            self.stoi = defaultdict(self._default_unk_index)
        else:
            self.stoi = defaultdict()
        self.stoi.update({tok: idx for idx, tok in enumerate(self.itos)})

    def _default_unk_index(self):
        return self.unk_index

    def __len__(self):
        return len(self.itos)

    def __eq__(self, other):
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        return True

    def extend(self, other, sort=False):
        words = sorted(other.itos) if sort else other.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.sto[w] = len(self.itos) - 1


__all__ = ['Vocab']