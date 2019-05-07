# -*- coding: utf-8 -*-

from spacy.tokens import Doc, Token
from spacy.lang import zh

from . import tokenizer


class JiebaTokenizer(object):
    def __init__(self, vocab, user_dict=None):
        self.vocab = vocab
        self.t = tokenizer.JiebaTokenizer(user_dict)

    def __call__(self, text):
        import sys; print('FUCK, text: %s' % text, file=sys.stderr)
        terms = self.t(text)
        words = [x.word for x in terms]
        spaces = [False] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


class Chinese(zh.Chinese):
    def make_doc(self, text):
        return self.tokenizer(text)
