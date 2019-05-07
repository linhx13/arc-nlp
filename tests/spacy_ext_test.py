# -*- coding: utf-8 -*-

import unittest
import sys

from spacy.lang import zh

sys.path.append("../")
from arcnlp.spacy_ext import JiebaTokenizer, Chinese


class TestChinese(unittest.TestCase):
    def setUp(self):
        self.zh_nlp = zh.Chinese()
        self.my_nlp = Chinese()
        self.my_nlp.tokenizer = JiebaTokenizer(self.my_nlp.vocab,
                                               user_dict="./user_dict.txt")

    def test_simple(self):
        doc = self.zh_nlp("这个乒乓球拍卖多少？")
        for x in doc:
            print(x)

    def test_user_dict(self):
        doc = self.my_nlp("这个乒乓球拍卖多少？")
        for x in doc:
            print(x)
            print(x._.pos)


if __name__ == '__main__':
    unittest.main()
