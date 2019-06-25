# -*- coding: utf-8 -*-


class Batch(object):
    """Defines a batch of examples along with its Fields.

    Attributes:
        batch_size: Number of examples in the batch.
        dataset: A reference to the dataset object the examples come from
            (which itself contains the dataset's Field objects).
        fields: The name of fields in the dataset.
        input_fields: The names of the fields that are used as input for model
        target_fields: The names of the fields that are used as targets during
            model training.

    Also store the tensor for each column in the batch as an attribute.
    """

    def __init__(self, data=None, dataset=None):
        """Create a Batch from a list of examples."""
        self.data = {}
        if data is not None:
            self.dataset = dataset
            self.fields = list(dataset.fields.keys())  # copy field names
            self.input_fields = sorted(k for k, v in dataset.fields.items()
                                       if v is not None and not v.is_target)
            self.target_fields = sorted(k for k, v in dataset.fields.items()
                                        if v is not None and v.is_target)
            for name, field in dataset.fields.items():
                if field is not None:
                    batch = [x[name] for x in data]
                    self.data[name] = field.process(batch)

    @classmethod
    def from_vars(cls, dataset, batch_size, train=None, **kwargs):
        """Create a Batch directly from a number of Variables."""
        batch = cls()
        batch.batch_size = batch_size
        batch.dataset = dataset
        batch.fields = list(dataset.fields.keys())
        for k, v in kwargs.items():
            setattr(batch, k, v)
        return batch

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, val):
        self.data[key] = val

    def __len__(self):
        return self.batch_size

    def _get_field_values(self, fields):
        if not fields:
            return None
        elif len(fields) == 1:
            return self.data[fields[0]]
        else:
            return {f: self.data[f] for f in fields}

    def as_training_data(self):
        return self._get_field_values(self.input_fields), \
            self._get_field_values(self.target_fields)

    # def __iter__(self):
    #     yield self._get_field_values(self.input_fields)
    #     yield self._get_field_values(self.target_fields)
