# -*- coding: utf-8 -*-

import unittest
import os

from spacy.lang import zh

from arcnlp.spacy_ext import JiebaTokenizer, Chinese

PWD = os.path.dirname(os.path.abspath(__file__))


class TestChinese(unittest.TestCase):
    def setUp(self):
        self.zh_nlp = zh.Chinese()
        self.my_nlp = Chinese()
        self.my_nlp.tokenizer = JiebaTokenizer(
            self.my_nlp.vocab, user_dict=os.path.join(PWD, "user_dict.txt"))

    def test_simple(self):
        doc = self.zh_nlp("这个乒乓球拍卖多少？")
        for x in doc:
            print(x)

    def test_user_dict(self):
        s = "这个乒乓球拍卖多少？"
        # for i in range(200000):
        #     ss = '%s%d' % (s, i)
        #     doc = self.my_nlp("这个乒乓球拍卖多少？")
        doc = self.my_nlp(s)
        for x in doc:
            print(x)
            print(x._.pos)


if __name__ == '__main__':
    unittest.main()
