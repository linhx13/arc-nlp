# -*- coding: utf-8 -*-

from .token import Token
from .tokenizer import Tokenizer
from .tokenizer import WhitespaceTokenizer, JiebaTokenizer, SpacyTokenizer
from .bert_tokenizer import BertTokenizer


def jieba_tokenizer(text):
    import jieba
    return jieba.luct(text)
