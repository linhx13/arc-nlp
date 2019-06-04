# -*- coding: utf-8 -*-

import io
from functools import partial

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
        self.make_example = partial(data.Example.from_list, fields=fields)
        gen_examples = partial(self.gen_examples, path=path, fields=fields,
                               encoding=encoding, separator=separator)
        if lazy:
            examples = gen_examples
        else:
            examples = list(gen_examples())
        super(SequenceTaggingDataset, self).__init__(examples, fields,
                                                     lazy=lazy, **kwargs)

    def gen_examples(self, path, fields, encoding, separator):
        columns = []
        with io.open(path, encoding=encoding) as fin:
            for line in fin:
                line = line.strip('\r\n')
                if line == "":
                    if columns:
                        yield self.make_example(columns)
                    columns = []
                else:
                    arr = line.split(separator)
                    if len(arr) != len(fields):
                        continue
                    for i, col in enumerate(arr):
                        if len(columns) < i + 1:
                            columns.append([])
                        columns[i].append(col)
            if columns:
                yield self.make_example(columns)

