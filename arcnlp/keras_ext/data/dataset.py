# -*- coding: utf-8 -*-

import six


class Dataset(object):
    """Define a dataset composed of Examples along with its fields.

    Attributes:
        sort_key (callable): A key to use for sorting dataset examples for
            batching.
        examples (list(Example)): The examples in this dataset.
        fields (dict[str, Field]): Contains the name of each column of field,
            together with the corresponding Field object. Two fields with same
            Field object will have a shared vocabulary.
    """

    sort_key = None

    def __init__(self, examples, fields, filter_pred=None):
        """Create a data from a list of Examples and Fields.

        Args:
            examples: List of Examples.
            fields (List(tuple(str, Filed))): The fields to use in this tuple.
                The string is a field name, and the Field is the associated
                field.
            filter_pred (callable or None): use onley examples for which
                filter_pred(example) is True, or use all examples if None.
                Default is None.
        """
        if filter_pred is not None:
            make_list = isinstance(examples, list)
            examples = six.moves.filter(filter_pred, examples)
            if make_list:
                examples = list(examples)
        self.examples = examples
        self.fields = dict(fields)
        # Unpack field tuples
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]

    def __getitem__(self, i):
        return self.exmples[i]

    def __len__(self):
        try:
            return len(self.examples)
        except TypeError:
            return 2 ** 32

    def __iter__(self):
        for x in self.examples:
            yield x

    def __getattr__(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)
