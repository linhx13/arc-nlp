# -*- coding: utf-8 -*-


class Batch(object):
    """Defines a batch of examples along with its Fields.

    Attributes:
        batch_size: Number of examples in the batch.
        dataset: A reference to the dataset object the examples come from
            (which itself contains the dataset's Field objects).
        feature_fields: The names of the fields that are used as input for the model
        target_fields: The names of the fields that are used as targets during
                       model training

    Also stores the Variable for each column in the batch as an attribute.
    """

    def __init__(self, data=None, dataset=None, device=None):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            self.dataset = dataset
            self.fields = dataset.fields.keys()  # copy field names
            self.feature_fields = [k for k, v in dataset.fields.items() if
                                   v is not None and not v.is_target]
            self.target_fields = [k for k, v in dataset.fields.items() if
                                  v is not None and v.is_target]

            for (name, field) in dataset.fields.items():
                if field is not None:
                    batch = [getattr(x, name) for x in data]
                    setattr(self, name, field.process(batch, device=device))

    def __len__(self):
        return self.batch_size

    def _get_field_values(self, fields):
        if len(fields) == 0:
            return None
        elif len(fields) == 1:
            return getattr(self, fields[0])
        else:
            return tuple(getattr(self, f) for f in fields)

    def __iter__(self):
        yield self._get_field_values(self.feature_fields)
        yield self._get_field_values(self.target_fields)
