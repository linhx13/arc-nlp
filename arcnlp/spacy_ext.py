# -*- coding: utf-8 -*-

from spacy.tokens import Doc, Token
from spacy.lang import zh

from . import tokenizers


class JiebaTokenizer(object):
    def __init__(self, vocab, user_dict=None):
        self.vocab = vocab
        self.user_dict = set()
        self.t = tokenizers.JiebaTokenizer()
        self.load_user_dict(user_dict)
        Token.set_extension("pos", default=None, force=True)

    def __call__(self, text):
        tokens = self.t.tokenize(text)
        words = [x.text for x in tokens]
        spaces = [False] * len(words)
        doc = Doc(self.vocab, words=words, spaces=spaces)
        for idx, token in enumerate(doc):
            token._.set('pos', tokens[idx].pos)
        return doc

    def __reduce__(self):
        args = (self.vocab, self.user_dict)
        return (self.__class__, args, None, None)

    def load_user_dict(self, user_dict):
        if not user_dict:
            return
        if isinstance(user_dict, str):
            with open(user_dict) as fin:
                user_dict = [line.strip('\r\n') for line in fin]
        self.user_dict.update(user_dict)
        self.t.load_user_dict(user_dict)


class Chinese(zh.Chinese):
    @classmethod
    def create(cls, user_dict=None):
        nlp = cls()
        nlp.tokenizer = JiebaTokenizer(vocab=nlp.vocab, user_dict=user_dict)
        return nlp

    def make_doc(self, text):
        return self.tokenizer(text)


def create_chinese(user_dict=None):
    if user_dict:
        nlp = Chinese()
        nlp.tokenizer = JiebaTokenizer(vocab=nlp.vocab, user_dict=user_dict)
    else:
        nlp = zh.Chinese()
    return nlp
