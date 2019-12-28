# -*- coding: utf-8 -*-

from typing import Dict, Sequence

from .sequence_tagging import SequenceTaggingDataHandler
from .. import Field


class Conll2000DataHandler(SequenceTaggingDataHandler):

    _COLUMNS = ['token', 'pos', 'chunk']

    def __init__(self,
                 token_fields: Dict[str, Field],
                 tag_column: str = "chunk",
                 feature_columns: Sequence[str] = None,
                 sparse_target=True,
                 sort_feature: str = None,):
        super(Conll2000DataHandler, self).__init__(
            columns=self._COLUMNS,
            token_fields=token_fields,
            tag_column=tag_column,
            feature_columns=feature_columns,
            sparse_target=sparse_target,
            sort_feature=sort_feature)
