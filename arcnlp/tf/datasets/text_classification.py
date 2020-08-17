from ..data import Dataset
from ..vocab import build_vocab_from_iterator


def build_vocab(data, transforms):
    tok_list = []
    for _, txt in data:
        tok_list.append(transforms(txt))
    return build_vocab_from_iterator(tok_list)


class TextClassificationDataset(Dataset):

    def __init__(self, data, vocab, transforms):
        """Initiate text-classification dataset.

        Arguments:
            data: a list of label and text tring tuple. label is an integer.
                [(label1, text1), (label2, text2), (label2, text3)]
            vocab: Vocabulary object used for dataset.
            transforms: a tuple of label and text string transforms.
        """

        super(TextClassificationDataset, self).__init__()
        self.data = data
        self.vocab = vocab
        self.transforms = transforms  # (label_transforms, tokens_transforms)

    def __getitem__(self, idx):
        label = self.data[idx][0]
        text = self.data[idx][1]
        if idx % 1000 == 0:
            print(idx)
        return (self.transforms[0](label), self.transforms[1](text))

    def __len__(self):
        return len(self.data)

    def get_vocab(self):
        return self.vocab
