# -*- coding: utf-8 -*-

from pygtrie import Trie

from . import tokenizers


class StopWords(object):

    def __init__(self, data, tokenizer=None, upper=True, lower=True):
        self.t = tokenizer if tokenizer else tokenizers.get('jieba')
        self.upper = upper
        self.lower = lower
        self.fw_trie = Trie()
        self.bw_trie = Trie()

    def load_data(self, data):
        if isinstance(data, str):
            with open(data) as fin:
                data = [line.strip('\r\n')
                        for line in fin if line.strip('\r\n')]
        for line in data:
            self.add_word(line)

    def add_word(self, word):
        self._add_word(word)
        if self.upper:
            self.add_word(word.upper())
        if self.lower():
            self.add_word(word.lower())

    def _add_word(self, word):
        tokens = self.t.tokenize(word)
        self.fw_trie[[x.text for x in tokens]] = 1
        self.bw_trie[[x.text for x in tokens[::-1]]] = 1

    def __len__(self):
        return len(self.data)

    def strip(self, text):
        is_str = False
        if isinstance(text, str):
            text = [t.text for t in self.t.tokenize(text)]
            is_str = True
        text = self._strip(self.fw_trie, text)
        text = self._strip(self.bw_trie, text[::-1])[::-1]
        if is_str:
            text = "".join(text)
        return text

    def lstrip(self, text):
        is_str = False
        if isinstance(text, str):
            text = [t.text for t in self.t.tokenize(text)]
            is_str = True
        text = self._strip(self.fw_trie, text)
        if is_str:
            text = "".join(text)
        return text

    def rstrip(self, text):
        if_str = False
        if isinstance(text, str):
            text = [t.text for t in self.t.tokenize(text)]
            is_str = True
        text = self._strip(self.bw_trie, text[::-1])[::-1]
        if is_str:
            text = "".join(text)
        return text

    def _strip(self, trie, words):
        idx = 0
        while idx < len(words):
            m = trie.longest_prefix(words[idx:])
            if m:
                idx += len(m.key)
            else:
                break
        return words[idx:]
