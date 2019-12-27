# -*- coding: utf-8 -*-

from typing import Dict, Sequence

from .sequence_tagging import SequenceTaggingDataHandler
from .. import Field


class Conll2000DataHandler(SequenceTaggingDataHandler):

    _COLUMNS = ['tokens', 'pos', 'chunk']

    def __init__(self,
                 token_fields: Dict[str, Field],
                 sort_feature: str = None,
                 tag_column: str = "chunk",
                 feature_columns: Sequence[str] = (),
                 sparse_target=True):
        super(Conll2000DataHandler, self).__init__(
            token_fields, sort_feature, self._COLUMNS, tag_column,
            feature_columns, sparse_target=sparse_target)
