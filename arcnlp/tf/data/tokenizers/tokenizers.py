import re
from functools import partial


__all__ = [
    'get_tokenizer',
    'split_tokenizer',
    'jieba_tokenizer',
    'spacy_tokenizer',
]



def split_tokenizer(text):
    return text.split()


def jieba_tokenizer(text):
    import jieba
    return jieba.lcut(text)


def _spacy_tokenize(text, spacy):
    return [tok.text for tok in spacy.tokenize(text)]


_spacy = None

def spacy_tokenizer(text, lang="en"):
    import spacy
    global _spacy
    if _spacy is None:
        _spacy = spacy.load(lang)
    return _spacy_tokenize(text, _spacy)


def get_tokenizer(tokenizer, lang='en'):
    if tokenizer is None:
        return split_tokenizer

    if callable(tokenizer):
        return tokenizer

    if tokenizer == 'spacy':
        import spacy
        spacy = spacy.load(lang)
        return partial(_spacy_tokenize, spacy=spacy)

    raise ValueError("Invalid tokenizer %s" % tokenizer)
