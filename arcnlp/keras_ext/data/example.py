# -*- coding: utf-8 -*-


class Example(object):
    """Defines a single training or test example.

    Stores each column of the example as an attribute.
    """

    @classmethod
    def from_dict(cls, data, fields):
        """Create an Example from a dict.

        Args:
            data: dict
            fields: dict[str, dict[str, Field]]
        """
        ex = cls()
        for key, fs in fields.items():
            if key not in data:
                raise ValueError("Specified key %s was not found in the "
                                 "input data" % key)
            if fs is not None:
                for n, f in fs.items():
                    setattr(ex, n, f.preprocess(data[key]))
        return ex

    # @classmethod
    # def from_json(cls, data, fields):
    #     return cls.from_dict(json.loads(data), fields)

    @classmethod
    def from_list(cls, data, fields):
        """Create an Example from a list.

        Args:
            data: list
            fields: list[dict[str, Field]]
        """
        ex = cls()
        for fs, val in zip(fields, data):
            if fs is not None:
                for n, f in fs.items():
                    setattr(ex, n, f.preprocess(val))
        return ex

    # @classmethod
    # def from_csv(cls, data, fields, field_to_index=None):
    #     if field_to_index is None:
    #         return cls.from_list(data, fields)
    #     else:
    #         assert(isinstance(fields, dict))
    #         data_dict = {f: data[idx] for f, idx in field_to_index.items()}
    #         return cls.from_dict(data_dict, fields)
