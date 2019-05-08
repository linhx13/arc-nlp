# -*- coding: utf-8 -*-

from spacy.tokens import Doc, Token
from spacy.lang import zh

from . import tokenizer


class JiebaTokenizer(object):
    def __init__(self, vocab, user_dict=None):
        self.vocab = vocab
        self.t = tokenizer.JiebaTokenizer(user_dict)
        Token.set_extension("pos", default=None, force=True)

    def __call__(self, text):
        terms = self.t(text)
        words = [x.word for x in terms]
        spaces = [False] * len(words)
        doc = Doc(self.vocab, words=words, spaces=spaces)
        for idx, token in enumerate(doc):
            token._.set('pos', terms[idx].flag)
        return doc


class Chinese(zh.Chinese):
    def make_doc(self, text):
        return self.tokenizer(text)


def create_chinese(user_dict=None):
    if user_dict:
        nlp = Chinese()
        nlp.tokenizer = JiebaTokenizer(vocab=nlp.vocab, user_dict=user_dict)
    else:
        nlp = zh.Chinese()
    return nlp
