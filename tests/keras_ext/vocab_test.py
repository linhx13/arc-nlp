# -*- coding: utf-8 -*-

import unittest
from collections import Counter
import pickle

from arcnlp.keras_ext.vocab import Vocab
import arcnlp.keras_ext.constants as C


class TestVocab(unittest.TestCase):

    def test_simple(self):
        text_data = ['hello', 'world', 'hello', 'nice', 'world', 'hi', 'world']
        counter = Counter(text_data)
        print('counter: %s' % counter)

        specials = [C.PAD_TOKEN, C.UNK_TOKEN, C.BOS_TOKEN, C.EOS_TOKEN]
        vocab = Vocab(counter, max_size=1, min_freq=2, specials=specials)
        print(vocab.stoi)
        print(vocab.itos)
        self.assertEqual(1, vocab.unk_index)

        for idx, tok in enumerate(specials):
            self.assertEqual(idx, vocab.stoi[tok])

        self.assertNotIn('hi', vocab.stoi)
        self.assertNotIn("nice", vocab.stoi)
        self.assertNotIn("hello", vocab.stoi)

        vocab_pkl = pickle.dumps(vocab)
        new_vocab = pickle.loads(vocab_pkl)
        self.assertEquals(vocab, new_vocab)
        print(new_vocab.stoi)
        print(new_vocab.itos)
