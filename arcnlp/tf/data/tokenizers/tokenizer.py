# -*- coding: utf-8 -*-

from typing import List
import re

from jieba import posseg

from .token import Token
from ...utils import get_spacy_model


class Tokenizer(object):

    def tokenize(self, text: str) -> List[Token]:
        raise NotImplementedError


class WhitespaceTokenizer(Tokenizer):

    def __init__(self, split_regex=r'\s+', lowercase=True):
        super(WhitespaceTokenizer, self).__init__()
        self.split_regex = split_regex
        self.lowercase = lowercase

    def tokenize(self, text: str) -> List[Token]:
        text = text.lower() if self.lowercase else text
        tokens = []
        for part in re.split(self.split_regex, text):
            tokens.append(Token(text=part))
        return tokens


class JiebaTokenizer(Tokenizer):

    def tokenize(self, text: str) -> List[Token]:
        return [Token(text=x[0], pos=x[1]) for x in posseg.cut(text)]


class SpacyTokenizer(Tokenizer):

    def __init__(self,
                 lang: str = 'en_core_web_sm'):
        super(SpacyTokenizer, self).__init__()
        self.nlp = get_spacy_model(lang, pos_tags=False, parse=False, ner=False)

    def tokenize(self, text: str) -> List[Token]:
        return [Token(text=t.text) for t in self.nlp(text) if not t.is_space]
