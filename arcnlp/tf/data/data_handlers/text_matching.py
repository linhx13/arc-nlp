# -*- coding: utf-8 -*-

from .data_handler import DataHandler
from .. import Field, LabelField, Example
from ..tokenizers import Tokenizer, JiebaTokenizer


class TextMatchingDataHandler(DataHandler):

    def __init__(self,
                 token_fields: Dict[str, Field],
                 tokenizer: Tokenizer = None,
                 sort_feature: str = None):
        self.tokenizer = tokenizer or JiebaTokenizer()
        