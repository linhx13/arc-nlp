# -*- coding: utf-8 -*-

from typing import Dict, Sequence

from .sequence_tagging import SequenceTaggingDataHandler
from .. import Field


class Conll2003DataHandler(SequenceTaggingDataHandler):

    _COLUMN_LABELS = ['tokens', 'pos', 'chunk', 'ner']

    def __init__(self,
                 token_fields: Dict[str, Field],
                 sort_feature: str,
                 tag_label: str = "ner",
                 feature_labels: Sequence[str] = (),
                 sparse_target=True):
        super(Conll2003DataHandler, self).__init__(
            token_fields, sort_feature, self._COLUMN_LABELS, tag_label,
            feature_labels, sparse_target=sparse_target
        )

    def _is_divider(self, line: str) -> bool:
        if line.strip() == "":
            return True
        else:
            first_token = line.split()[0]
            return first_token == "-DOCSTART-"
