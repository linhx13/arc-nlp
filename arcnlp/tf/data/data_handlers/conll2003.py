# -*- coding: utf-8 -*-

from typing import Dict, Sequence

from .sequence_tagging import SequenceTaggingDataHandler
from .. import Field


class Conll2003DataHandler(SequenceTaggingDataHandler):

    _COLUMNS = ['token', 'pos', 'chunk', 'ner']

    def __init__(self,
                 token_fields: Dict[str, Field],
                 tag_column: str = "ner",
                 feature_columns: Sequence[str] = None,
                 sparse_target=True,
                 sort_feature: str = None):
        super(Conll2003DataHandler, self).__init__(
            token_fields=token_fields,
            columns=self._COLUMNS,
            tag_column=tag_column,
            feature_columns=feature_columns,
            sparse_target=sparse_target,
            sort_feature=sort_feature)

    def _is_divider(self, line: str) -> bool:
        if line.strip() == "":
            return True
        else:
            first_token = line.split()[0]
            return first_token == "-DOCSTART-"
