# -*- coding: utf-8 -*-

import unittest
import os

from arcnlp.tokenizer import JiebaTokenizer

PWD = os.path.dirname(os.path.abspath(__file__))


class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = JiebaTokenizer(
            user_dict=os.path.join(PWD, 'user_dict.txt'))

    def test_simple(self):
        s = "这个乒乓球拍卖多少？"
        for i in range(200000):
            ss = '%s%d' % (s, i)
            for x in self.tokenizer(ss):
                pass


if __name__ == '__main__':
    unittest.main()

