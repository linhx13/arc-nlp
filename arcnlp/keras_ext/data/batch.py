# -*- coding: utf-8 -*-


class Batch(object):
    """Defines a batch of examples along with its Fields.

    Attributes:
        batch_size: Number of examples in the batch.
        dataset: A reference to the dataset object the examples come from
            (which itself contains the dataset's Field objects).
        fields: The name of fields in the dataset.
    """

    def __init__(self, data=None, dataset=None):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            self.dataset = dataset
            self.fields = list(dataset.fields.keys())  # copy field names
            for (name, field) in dataset.fields.items():
                if field is not None:
                    batch = [getattr(x, name) for x in data]
                    setattr(self, name, field.process(batch))

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

    def __len__(self):
        return self.batch_size
