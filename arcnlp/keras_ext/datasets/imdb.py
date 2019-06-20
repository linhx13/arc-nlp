# -*- coding: utf-8 -*-

import os
import glob
import io

from .. import data


class IMDB(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_fields, label_fields, **kwargs):
        """Create an IMDB dataset.

        Args:
            path: Path to the dataset's top level directory
            text_fields: Dict[str, Field]
            labe_fields: Dict[str, Field]
            kwargs
        """
        fields = [text_fields, label_fields]
        examples = []

        for label in ['pos', 'neg']:
            for fname in glob.iglob(os.path.join(path, label, '*.txt')):
                with io.open(fname, 'r', encoding='utf-8') as f:
                    text = f.readline()
                examples.append(data.Example.from_list([text, label], fields))

        super(IMDB, self).__init__(examples, fields, **kwargs)
