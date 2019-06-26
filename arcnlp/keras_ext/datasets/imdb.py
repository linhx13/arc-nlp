# -*- coding: utf-8 -*-

import os
import glob
import io

from .. import data


class IMDB(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, **kwargs):
        """Create an IMDB dataset.

        Args:
            path: Path to the dataset's top level directory
            text_fields: Dict[str, Field]
            **kwargs:
        """
        fields = [('text', text_field), ('label', label_field)]
        examples = []

        for label in ['pos', 'neg']:
            for fname in glob.iglob(os.path.join(path, label, '*.txt')):
                with io.open(fname, 'r', encoding='utf-8') as f:
                    text = f.readline()
                examples.append(data.Example.from_list([text, label], fields))

        super(IMDB, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, root='.data',
               train='train', test='test', **kwargs):
        """Create dataset objects for splits of the IMDB dataset.

        Args:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for the label.
            root: Root dataset storage directory. Default is ".data".
            train: The directory that contains the training examples.
            test: The directory that contains the test examples.
            **kwargs: Passed to the splits method of Dataset.
        """
        return super(IMDB, cls).splits(
            root=root, text_field=text_field, label_field=label_field,
            train=train, validation=None, test=test, **kwargs)
