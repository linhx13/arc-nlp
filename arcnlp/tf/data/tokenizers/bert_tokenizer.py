# -*- coding: utf-8 -*-

import os
import codecs

os.environ['TF_KERAS'] = '1'

from keras_bert import Tokenizer


class BertTokenizer(Tokenizer):

    def __init__(self, vocab_file, **kwargs):
        token_dict = {}
        with codecs.open(vocab_file, 'r', encoding='utf-8') as fin:
            for line in fin:
                token = line.strip('\r\n')
                token_dict[token] = len(token_dict)
        super(BertTokenizer, self).__init__(token_dict, **kwargs)

    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R
