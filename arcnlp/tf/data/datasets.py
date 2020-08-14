from typing import Iterable
from collections import Counter
import tensorflow as tf

class Vocab():
    pass


class Dataset(object):
    def __init__(self):
        pass

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class TextClassificationDataset(Dataset):

    def __init__(self, data, vocab, transforms):
        super(TextClassificationDataset, self).__init__()
        self.data = data
        self.vocab = vocab
        self.transforms = transforms

    def __getitem__(self, idx):
        label = self.data[idx][0]
        text = self.data[idx][1]
        return (self.transforms[0](label), self.transforms[1](text))

    def __len__(self):
        return len(self.data)


def sequential_transforms(*transforms):
    def func(text_input):
        for transform in transforms:
            text_input = transform(text_input)
        return text_input

    return func


def vocab_func(vocab):
    def func(tok_iter):
        return [vocab[tok] for tok in tok_iter]

    return func


def totensor(dtype):
    def func(ids_list):
        return tf.convert_to_tensor(ids_list, dtype=dtype)

    return func


class TextClassificationDataHandler:

    def __init__(self, text_field, label_field):
        self.text_field = text_field
        self.label_field = label_field

    def build_vocabs(self, *raw_datasets):
        text_counter, label_counter = Counter(), Counter()
        for dataset in raw_datasets:
            for example in dataset:
                if self.text_field.use_vocab:
                    self.text_field.count_vocab(example['text'], text_counter)
                if self.label_field.use_vocab:
                    self.text_field.count_vocab(example['label'], label_counter)
        if self.text_field.use_vocab:
            self.text_field.vocab = Vocab(text_counter)
        if self.label_field.use_vocab:
            self.label_field.vocab = Vocab(label_counter)

    def build_raw_dataset(self, path) -> Iterable:
        """ read examples from path """
        pass

    def build_dataset(self, path):
        text_transform = []
        text_transform = sequential_transforms(self.tokenizer)
        raw_dataset = self.build_raw_dataset(path)
        self.build_vocab() # TODO:

        text_transform = sequential_transforms(
            text_transform, vocab_func(self.text_vocab), totensor(dtype='int32'))
        label_transform = sequential_transforms(
            vocab_func(self.label_vocab), totensor(dtype='int32'))
        return TextClassificationDataset(
            raw_dataset, (label_transform, text_transform))

    def build_data_iter(self, dataset, batch_size=32, train=True) -> tf.data.Dataset:
        pass


class DataLoader(tf.keras.utils.Sequence):
    pass


class DataLoader:

    def __init__(self, dataset):
        pass
