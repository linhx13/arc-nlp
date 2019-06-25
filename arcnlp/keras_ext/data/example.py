# -*- coding: utf-8 -*-

from typing import Mapping, Any


class Example(Mapping[str, Any]):
    """Defines a single training or test example.

    Stores each column of the example as an attribute.
    """

    def __init__(self, data=None):
        self.data = data if data is not None else {}

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, val):
        self.data[key] = val

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    @classmethod
    def from_dict(cls, data, fields):
        """Create an Example from a dict.

        Args:
            data: dict
            fields: list[tuple[str, Union[Field, Dict[str, Field]]]]
        """
        ex = cls()
        for key, fs in fields.items():
            if key not in data:
                raise ValueError("Specified key %s was not found in the "
                                 "input data" % key)
            if fs is not None:
                if isinstance(fs, dict):
                    for n, f in fs.items():
                        ex['%s.%s' % (key, n)] = f.preprocess(data[key])
                else:
                    ex[key] = fs.preprocess(data[key])
        return ex

    # @classmethod
    # def from_json(cls, data, fields):
    #     return cls.from_dict(json.loads(data), fields)

    @classmethod
    def from_list(cls, data, fields):
        """Create an Example from a list.

        Args:
            data: list
            fields: list[tuple[str, Union[Field, Dict[str, Field]]]]
        """
        ex = cls()
        for (key, fs), val in zip(fields, data):
            if fs is not None:
                if isinstance(fs, dict):
                    for n, f in fs.items():
                        ex['%s.%s' % (key, n)] = f.preprocess(val)
                else:
                    ex[key] = fs.preprocess(val)
        return ex

    # @classmethod
    # def from_csv(cls, data, fields, field_to_index=None):
    #     if field_to_index is None:
    #         return cls.from_list(data, fields)
    #     else:
    #         assert(isinstance(fields, dict))
    #         data_dict = {f: data[idx] for f, idx in field_to_index.items()}
    #         return cls.from_dict(data_dict, fields)
