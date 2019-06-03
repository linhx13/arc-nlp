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
        examples = self.gen_examples(path, fields, encoding, separator)
        if not lazy:
            examples = list(examples)
        super(SequenceTaggingDataset, self).__init__(examples, fields,
                                                     **kwargs)

    def gen_examples(self, path, fields, encoding, separator, lazy):
        while True:
            columns = []
            with io.open(path, encoding=encoding) as input_file:
                for line in input_file:
                    line = line.strip('\r\n')
                    if line == "":
                        if columns:
                            yield data.Example.from_list(columns, fields)
                        columns = []
                    else:
                        for i, column in enumerate(line.split(separator)):
                            if len(columns) < i + 1:
                                columns.append([])
                            columns[i].append(column)
                if columns:
                    yield data.Example.from_list(columns, fields)
            # FIXME: lazy logic
            if not lazy:
                break
