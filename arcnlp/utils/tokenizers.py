# -*- coding: utf-8 -*-

from typing import Union, Iterable, List

from jieba import posseg
from pygtrie import Trie


class Token(object):
    def __init__(self, text: str, pos: str):
        self.text = text
        self.pos = pos

    def __repr__(self):
        return "Token<text:%s,pos:%s>" % (self.text, self.pos)


class JiebaTokenizer(object):
    def __init__(self, user_dict: Union[str, Iterable] = None):
        self.t = posseg.POSTokenizer()
        self.t.initialize()
        self.trie = Trie()
        if user_dict:
            self.load_user_dict(user_dict)

    def load_user_dict(self, user_dict: Union[str, Iterable] = None):
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
                value = [x for x in arr[1].strip().split()]
                assert len(key) == sum(len(x) for x in value)
            else:
                value = [key]
            seg_dict[key] = value
            self.t.tokenizer.add_word(key, 100)
        # jieba一定会切分英文串，对于英文还需要trie树来合并一下
        for key, value in seg_dict.items():
            self.trie[self.t.tokenizer.lcut(key)] = value

    def tokenize(self, s: str) -> List[Token]:
        res = []
        term_list = list(self.t.cut(s))
        text_list = [x.word for x in term_list]
        idx = 0
        while idx < len(text_list):
            m = self.trie.longest_prefix(text_list[idx:])
            if m:
                res.extend(Token(x, 'nz') for x in m.value)
                idx += len(m.key)
            else:
                res.append(Token(term_list[idx].word, term_list[idx].flag))
                idx += 1
        return res


def get(name, user_dict=None, **kwargs):
    if name == "jieba":
        return JiebaTokenizer(user_dict=user_dict, **kwargs)
    else:
        raise ValueError("Unsupported tokenizer type: %s" % name)
