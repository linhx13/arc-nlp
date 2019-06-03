# -*- coding: utf-8 -*-

import json

import six


class Example(object):
    """Defines a single training or test example.

    Stores each column of the example as an attribute.
    """

    @classmethod
    def from_dict(cls, data, fields):
        ex = cls()
        for key, vals in fields.items():
            if key not in data:
                raise ValueError("Specified key %s was not found in the "
                                 "input data" % key)
            if vals is not None:
                if not isinstance(vals, list):
                    vals = [vals]
                for name, field in vals:
                    setattr(ex, name, field.preprocess(data[key]))
        return ex

    @classmethod
    def from_json(cls, data, fields):
        return cls.from_dict(json.loads(data), fields)

    @classmethod
    def from_list(cls, data, fields):
        ex = cls()
        for (name, field), val in zip(fields, data):
            if field is not None:
                if isinstance(val, six.string_types):
                    val = val.rstrip('\r\n')
                if isinstance(name, tuple):
                    for n, f in zip(name, field):
                        setattr(ex, n, f.preprocess(val))
                else:
                    setattr(ex, name, field.preprocess(val))
        return ex

    @classmethod
    def from_csv(cls, data, fields, field_to_index=None):
        if field_to_index is None:
            return cls.from_list(data, fields)
        else:
            assert(isinstance(fields, dict))
            data_dict = {f: data[idx] for f, idx in field_to_index.items()}
            return cls.from_dict(data_dict, fields)
