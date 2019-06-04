# -*- coding: utf-8 -*-

from six.moves import filter


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

    def __init__(self, examples, fields, filter_pred=None, lazy=False):
        """Create a data from a list of Examples and Fields.

        Args:
            examples: List of Examples or a callable which returns a generator.
            fields (List(tuple(str, Filed))): The fields to use in this tuple.
                The string is a field name, and the Field is the associated
                field.
            filter_pred (callable or None): use onley examples for which
                filter_pred(example) is True, or use all examples if None.
                Default is None.
            lazy: If this is true, `examples` should a callable which returns
                a examples generator, otherwise `examples` shoule be a list.
                Default is False.
        """
        self.lazy = lazy
        if not lazy:
            if filter_pred is not None:
                examples = filter(filter_pred, examples)
            examples = list(examples)
        else:
            self.filter_pred = filter_pred
        self._examples = examples
        self.fields = dict(fields)
        # Unpack field tuples
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]

    @property
    def examples(self):
        if self.lazy:
            examples = self._examples()
            assert not isinstance(examples, list)
            if self.filter_pred is not None:
                examples = filter(self.filter_pred, examples)
            return examples
        else:
            return self._examples

    def __getitem__(self, i):
        examples = self.examples
        if hasattr(examples, '__getitem__'):
            return examples[i]
        else:
            raise TypeError('Examples type %s does not support __getitem__'
                            % type(examples))

    def __len__(self):
        examples = self.examples
        try:
            return len(examples)
        except TypeError:
            raise TypeError("Examples type %s does not support __len__"
                            % type(examples))

    def __iter__(self):
        for x in self.examples:
            yield x

    def __getattr__(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)
