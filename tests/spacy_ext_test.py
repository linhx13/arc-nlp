# -*- coding: utf-8 -*-

import unittest
import sys

from spacy.lang import zh

sys.path.append("../")
from arcnlp.spacy_ext import JiebaTokenizer, Chinese


class TestChinese(unittest.TestCase):
    def test_simple(self):
        # nlp = Chinese()
        # nlp.tokenizer = JiebaTokenizer(nlp.vocab)
        nlp = zh.Chinese()
        doc = nlp("这个乒乓球拍卖多少？")
        for x in doc:
            print(x)

    def test_user_dict(self):
        # nlp = zh.Chinese()
        nlp = Chinese()
        nlp.tokenizer = JiebaTokenizer(nlp.vocab, user_dict='./user_dict.txt')
        doc = nlp("这个乒乓球拍卖多少？")
        for x in doc:
            print(x)
            print(x._.pos)


if __name__ == '__main__':
    unittest.main()
