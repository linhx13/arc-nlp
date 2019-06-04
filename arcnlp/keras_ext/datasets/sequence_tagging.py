# -*- coding: utf-8 -*-

import io

from .. import data


class SequenceTaggingDataset(data.Dataset):
    """Defines a dataset for sequence tagging. Examples in this dataset
    contain paired lists -- paired list of words and tags.
    """

    @staticmethod
    def sort_key(example):
        # TODO:
        return 0

    def __init__(self, path, fields, encoding="utf-8", separator="\t",
                 lazy=False, **kwargs):
        self.path = path
        self.fields = fields
        self.encoding = encoding
        self.separator = separator
        if lazy:
            examples = self.gen_examples
        if not lazy:
            examples = list(self.gen_examples())
        super(SequenceTaggingDataset, self).__init__(examples, fields,
                                                     lazy=lazy, **kwargs)

    def gen_examples(self):
        columns = []
        with io.open(self.path, encoding=self.encoding) as fin:
            for line in fin:
                line = line.strip('\r\n')
                if line == "":
                    if columns:
                        yield data.Example.from_list(columns, self.fields)
                    columns = []
                else:
                    arr = line.split(self.separator)
                    if len(arr) != len(self.fields):
                        continue
                    for i, col in enumerate(arr):
                        if len(columns) < i + 1:
                            columns.append([])
                        columns[i].append(col)
            if columns:
                yield data.Example.from_list(columns, self.fields)

