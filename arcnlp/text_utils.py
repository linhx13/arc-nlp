# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

import re
import itertools
from zhon import hanzi
import logging
import six
from six import iteritems, unichr


logger = logging.getLogger(__name__)

if not six.PY2:
    unicode = str


def to_utf8(text, encoding='utf-8', errors='strict'):
    """Convert a string (unicode or bytestring in `encoding`), to bystring in
    utf-8 encoding. """
    if isinstance(text, unicode):
        return text.encode('utf-8')
    else:
        return unicode(text, encoding, errors=errors).encode('utf-8')


def to_unicode(text, encoding='utf-8', errors='strict'):
    """Convert a string (bytesstring in `encoding` or unicode) to unicode. """
    if isinstance(text, unicode):
        return text
    else:
        return unicode(text, encoding, errors=errors)


__english_periods = u'\r|\n|\?!|!|\?|\. '
__three_periods = u'？！”|。’”|！’”|……”'
__two_periods = u'。”|！”|？”|；”|？！|……'
__one_periods = u'！|？|｡|。'

__periods_pat = re.compile(u'(%s)' % '|'.join(
    [__english_periods, __three_periods, __two_periods, __one_periods]))


ENGLISH_SENTENCE_END = u'\r|\n|\?!\s*|!\s*|\?\s*|\. '
CHINESE_SENTENCE_END = hanzi._sentence_end
SENTENCE_END = u'(?:%s)' % \
    ('|'.join([ENGLISH_SENTENCE_END, CHINESE_SENTENCE_END, '$']))
SENTENCE = u'.+?%s' % SENTENCE_END


def split_sentences(text):
    ''' Split text into sentences.

    Args:
      text: unicode string needed to split
    Returns:
      a list of sentences
    '''
    return re.findall(SENTENCE, text)


def str_half2full(text):
    ''' Convert text from half-width characters to full-width characters. '''
    def _conv(c):
        code = ord(c)
        if code == 0x0020:
            code = 0x3000
        elif 0x0021 <= code <= 0x007e:
            code += 0xfee0
        return unichr(code)
    return ''.join(_conv(c) for c in text)


def str_full2half(text):
    '''Conver text from full-width characters to full-width characters. '''
    def _conv(c):
        code = ord(c)
        if code == 0x3000:
            code = 0x0020
        elif 0xff01 <= code <= 0xff5e:
            code -= 0xfee0
        return unichr(code)
    return ''.join(_conv(c) for c in text)


FH_SPACE = FHS = ((u"　", u" "),)
FH_NUM = FHN = (
    (u"０", u"0"), (u"１", u"1"), (u"２", u"2"), (u"３", u"3"), (u"４", u"4"),
    (u"５", u"5"), (u"６", u"6"), (u"７", u"7"), (u"８", u"8"), (u"９", u"9"),
)
FH_ALPHA = FHA = (
    (u"ａ", u"a"), (u"ｂ", u"b"), (u"ｃ", u"c"), (u"ｄ", u"d"), (u"ｅ", u"e"),
    (u"ｆ", u"f"), (u"ｇ", u"g"), (u"ｈ", u"h"), (u"ｉ", u"i"), (u"ｊ", u"j"),
    (u"ｋ", u"k"), (u"ｌ", u"l"), (u"ｍ", u"m"), (u"ｎ", u"n"), (u"ｏ", u"o"),
    (u"ｐ", u"p"), (u"ｑ", u"q"), (u"ｒ", u"r"), (u"ｓ", u"s"), (u"ｔ", u"t"),
    (u"ｕ", u"u"), (u"ｖ", u"v"), (u"ｗ", u"w"), (u"ｘ", u"x"), (u"ｙ", u"y"),
    (u"ｚ", u"z"),
    (u"Ａ", u"A"), (u"Ｂ", u"B"), (u"Ｃ", u"C"), (u"Ｄ", u"D"), (u"Ｅ", u"E"),
    (u"Ｆ", u"F"), (u"Ｇ", u"G"), (u"Ｈ", u"H"), (u"Ｉ", u"I"), (u"Ｊ", u"J"),
    (u"Ｋ", u"K"), (u"Ｌ", u"L"), (u"Ｍ", u"M"), (u"Ｎ", u"N"), (u"Ｏ", u"O"),
    (u"Ｐ", u"P"), (u"Ｑ", u"Q"), (u"Ｒ", u"R"), (u"Ｓ", u"S"), (u"Ｔ", u"T"),
    (u"Ｕ", u"U"), (u"Ｖ", u"V"), (u"Ｗ", u"W"), (u"Ｘ", u"X"), (u"Ｙ", u"Y"),
    (u"Ｚ", u"Z"),
)
FH_PUNCTUATION = FHP = (
    (u"．", u"."), (u"，", u","), (u"！", u"!"), (u"？", u"?"), (u"”", u'"'),
    (u"’", u"'"), (u"‘", u"`"), (u"＠", u"@"), (u"＿", u"_"), (u"：", u":"),
    (u"；", u";"), (u"＃", u"#"), (u"＄", u"$"), (u"％", u"%"), (u"＆", u"&"),
    (u"（", u"("), (u"）", u")"), (u"‐", u"-"), (u"＝", u"="), (u"＊", u"*"),
    (u"＋", u"+"), (u"－", u"-"), (u"／", u"/"), (u"＜", u"<"), (u"＞", u">"),
    (u"［", u"["), (u"￥", u"\\"), (u"］", u"]"), (u"＾", u"^"), (u"｛", u"{"),
    (u"｜", u"|"), (u"｝", u"}"), (u"～", u"~"),
)

FH_ASCII = HAC = lambda: ((fr, to) for m in (FH_ALPHA, FH_NUM, FH_PUNCTUATION)
                          for fr, to in m)

HF_SPACE = HFS = ((u" ", u"　"),)
HF_NUM = HFN = lambda: ((h, z) for z, h in FH_NUM)
HF_ALPHA = HFA = lambda: ((h, z) for z, h in FH_ALPHA)
HF_PUNCTUATION = HFP = lambda: ((h, z) for z, h in FH_PUNCTUATION)
HF_ASCII = ZAC = lambda: ((h, z) for z, h in FH_ASCII())


def convert_fh(text, *maps, **ops):
    """ Convert between full-width and half-width characters.

    Args:
      text: unicode string need to convert
      maps: conversion maps
      skip: skip out of character. In a tuple or string
    Returns:
      converted unicode string
    """
    if "skip" in ops:
        skip = ops["skip"]
        if isinstance(skip, str):
            skip = tuple(skip)

        def replace(text, fr, to):
            return text if fr in skip else text.replace(fr, to)
    else:
        def replace(text, fr, to):
            return text.replace(fr, to)

    for m in maps:
        if callable(m):
            m = m()
        elif isinstance(m, dict):
            m = m.items()
        for fr, to in m:
            text = replace(text, fr, to)
    return text


def find_ngrams(seq, n):
    return list(six.moves.zip(*[itertools.islice(seq, i, None) for i in range(n)]))


def extract_ngrams(sequence, ngram_range, sep=None):
    sequence = list(sequence)
    it = itertools.chain \
        .from_iterable(find_ngrams(sequence, n)
                       for n in range(ngram_range[0], ngram_range[1]+1))
    if sep is not None:
        it = six.moves.map(sep.join, it)
    return list(it)


URL = u'(?:(?:https?|ftp|file)://|www\.|ftp\.)[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'
EMAIL = u'''[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*'''
PHONE = u'(?<!\d)(?:\+?[0-9]{2,3}-)?(?:[0-9]{2,4}-)?[0-9]{7,8}(?:-[0-9]{1,4})?(?!\d)'
MOBILE = u'(?<!\d)(?:\+?[0-9]{2,3}-)?1(?:3[0-9]|4[57]|5[0-35-9]|7[0135678]|8[0-9])\d{8}(?!\d)'
ZIP_CODE = u'(?<!\d)[1-9]{1}\d{5}(?!\d)'
QQ = u'(?:(?:QQ|qq).{5,10})[1-9][0-9]{4,}(?!\d)'


ENTITY_MAPPING = {
    URL: '<url>', EMAIL: '<email>', PHONE: '<phone>', MOBILE: '<mobile>',
    ZIP_CODE: '<zip_code>', QQ: '<qq>'
}


def simple_preprocess(text, *maps, **ops):
    ''' Simpole preprocess.

    Args:
      text: unicode string to process
      maps: conversion maps
      ops: operations to do. Supported: trim_space, t2s, full2half, lower.

    Returns:
      procesed string.
    '''
    for m in maps:
        for fr, to in iteritems(m):
            text = re.sub(fr, to, text)
    if not text:
        return text
    if ops.get('trim_space', False):
        text = re.sub(u'\s{2,}', ' ', text)
    if ops.get('t2s', False):
        import opencc
        text = opencc.convert(text)
    if ops.get('full2half', False):
        text = str_full2half(text)
    if ops.get('lower', False):
        text = text.lower()
    return text


def merge_segmented_entities(words, entity_list):
    if isinstance(words, six.string_types):
        s = words
    else:
        s = ' '.join(words)
    res = []
    cur = []
    for t in s:
        if t == '<':
            res.extend(cur)
            cur = [t]
        elif t == '>':
            cur.append(t)
            e = re.sub(u'\s+', '', ''.join(cur))
            if e in entity_list:
                if len(res) > 0 and res[-1][-1] != ' ':
                    res.append(' ')
                res.append(e)
            else:
                res.extend(cur)
            cur = []
        else:
            cur.append(t)
    res.extend(cur)
    res = ''.join(res)
    if isinstance(words, six.string_types):
        return res
    else:
        return res.split(' ')


def clean_html_tags(raw_html):
  cleantext = re.sub('<.*?>', '', raw_html)
  return cleantext


if __name__ == '__main__':
    text = u"成田空港—【ＪＲ特急成田エクスプレス号・横浜行，2站】—東京—【ＪＲ新幹線はやぶさ号・新青森行,6站 】—新青森—【ＪＲ特急スーパー白鳥号・函館行，4站 】—函館"
    print(convert_fh(text, FH_ASCII,
                     {u"【": u"[", u"】": u"]", u",": u"，", u".": u"。",
                      u"?": u"？", u"!": u"！"},
                     spit="，。？！“”"))

    text = u'你好，我的邮箱是123@bbb.com，手机是13811111111'
    text = simple_preprocess(text, ENTITY_MAPPING, lower=True,
                             full2half=True, t2s=True, trim_space=True)
    import jieba
    words = jieba.cut(text)
    words = merge_segmented_entities(words, ENTITY_MAPPING.values())
    for w in words:
        print(w)
