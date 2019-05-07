# -*- coding: utf-8 -*-

from jieba import posseg
from pygtrie import Trie


class JiebaTokenizer(object):
    def __init__(self, user_dict=None):
        self.t = posseg.POSTokenizer()
        self.trie = Trie()
        if user_dict:
            self.load_user_dict(user_dict)

    def load_user_dict(self, user_dict):
        if isinstance(user_dict, str):
            with open(user_dict) as fin:
                user_dict = [line.strip('\r\n') for line in fin]
        seg_dict = {}
        for line in user_dict:
            line = line.strip()
            if not line:
                continue
            arr = line.split('=', 1)
            key = arr[0].strip()
            if len(arr) > 1 and len(arr[1].strip()) > 0:
                value = [x.strip() for x in arr[1].strip().split()]
            else:
                value = [key]
            seg_dict[key] = value
            self.t.tokenizer.add_word(key, 100)
        for key, value in seg_dict.items():
            self.trie[self.t.tokenizer.lcut(key)] = value

    def __call__(self, s):
        res = []
        term_list = list(self.t.cut(s))
        text_list = [x.word for x in term_list]
        idx = 0
        while idx < len(text_list):
            m = self.trie.longest_prefix(text_list[idx:])
            if m:
                res.extend([posseg.pair(x, 'nz') for x in m.value])
                idx += len(m.key)
            else:
                res.append(term_list[idx])
                idx += 1
        return res
