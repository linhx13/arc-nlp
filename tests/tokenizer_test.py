# -*- coding: utf-8 -*-

import unittest
import sys

sys.path.append("../")
from arcnlp.tokenizer import JiebaTokenizer


class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = JiebaTokenizer(user_dict='./user_dict.txt')

    def test_simple(self):
        s = "这个乒乓球拍卖多少？"
        for i in range(200000):
            ss = '%s%d' % (s, i)
            for x in self.tokenizer(ss):
                pass


if __name__ == '__main__':
    unittest.main()