# -*- coding: utf-8 -*-

from six.moves import filter

from .example import Example


class Dataset(object):
    sort_key = None

    def __init__(self, examples, fields, filter_pred=None):
        if filter_pred is not None:
            make_list = isinstance(examples, list)
            examples = filter(filter_pred, examples)
            if make_list:
                examples = list(examples)
        self.examples = examples
        self.fields = dict(fields)
        # TODO: Unpack field tuples
        # for n, f in list(self.fields.items()):
        #     if isinstance(n, tuple):
        #         self.fields.update(zip(n, f))
        #         del self.fields[n]

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


class LazyDataset(object):
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
        self.length = None
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
        if self.length is None:
            self.length = sum(1 for _ in self.examples)
        return self.length

    def __iter__(self):
        for x in self.examples:
            yield x

    def __getattr__(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)


class TabularDataset(Dataset):
    """Defines a Dataset of columns from list of list or list of dict."""

    def __init__(self, data, format, fields, **kwargs):
        format = format.lower()
        make_example = {
            "list": Example.from_list, "dict": Example.from_dict
        }[format]

        examples = [make_example(item, fields) for item in data]

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)
        else:
            fields, field_list = [], fields
            for field in field_list:
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)
        super(TabularDataset, self).__init__(examples, fields, **kwargs)
