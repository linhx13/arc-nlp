# -*- coding: utf-8 -*-

import unittest
import os

from arcnlp.tokenizers import JiebaTokenizer

PWD = os.path.dirname(os.path.abspath(__file__))


class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = JiebaTokenizer(
            user_dict=os.path.join(PWD, 'user_dict.txt'))

    def test_simple_perf(self):
        s = "这个乒乓球拍卖多少？"
        for i in range(200000):
            ss = '%s%d' % (s, i)
            for x in self.tokenizer.tokenize(ss):
                pass

    def test_simple(self):
        s = "这个乒乓球拍卖多少？"
        print(self.tokenizer.tokenize(s))


if __name__ == '__main__':
    unittest.main()

