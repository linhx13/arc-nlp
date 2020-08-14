# -*- coding: utf-8 -*-


class Dataset(object):

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class OldDataset(object):
    """Defines a dataset composed of Examples along with its Fields.

    Attributes:
        sort_key (callable): A key to use for sorting dataset examples for batching
            together examples with similar lengths to minimize padding.
        examples (list(Example)): The examples in this dataset.
        fields (dict[str, Field]): Contains the name of each column or field, together
            with the corresponding Field object. Two fields with the same Field object
            will have a shared vocabulary.
    """


    sort_key = None

    def __init__(self, examples, fields):
        """Create a dataset from a list of Examples and Fields.

        Arguments:
            examples: List of Examples.
            fields (List(tuple(str, Field))): The Fields to use in this tuple. The
                string is a field name, and the Field is the associated field.
        """
        self.examples = examples
        self.fields = dict(fields)
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]

    def __getitem__(self, i):
        return self.examples[i]

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
