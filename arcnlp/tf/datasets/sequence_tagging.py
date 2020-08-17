from ..data import Dataset
from ..vocab import build_vocab_from_iterator


def create_data_from_iob(data_path, separator="\t"):
    with open(data_path, encoding="utf-8") as input_file:
        columns = []
        for line in input_file:
            line = line.strip()
            if line == "":
                if columns:
                    yield columns
                columns = []
            else:
                for i, column in enumerate(line.split(separator)):
                    if len(columns) < i + 1:
                        columns.append([])
                    columns[i].append(column)
        if len(columns) > 0:
            yield columns


def build_vocab(data):
    total_columns = len(data[0])
    data_list = [[] for _ in range(total_columns)]
    vocabs = []

    for line in data:
        for idx, col in enumerate(line):
            data_list[idx].append(col)

    for it in data_list:
        vocabs.append(build_vocab_from_iterator(it))

    return vocabs


class SequenceTaggingDataset(Dataset):

    def __init__(self, data, vocabs, transforms):
        """Initiate sequence tagging dataset.
        Arguments:
            data: a list of word and its respective tags. Example:
                [[word, POS, dep_parsing label, ...]]
            vocabs: a list of vocabularies for its respective tags.
                The number of vocabs must be the same as the number of columns
                found in the data.
            transforms: a list of string transforms for words and tags.
                The number of transforms must be the same as the number of columns
                    found in the data.
        """

        super(SequenceTaggingDataset, self).__init__()
        self.data = data
        self.vocabs = vocabs
        self.transforms = transforms

        if len(self.data[0])!= len(self.vocabs):
            raise ValueError("vocabs must have the same number of columns "
                             "as the data")

    def __getitem__(self, idx):
        cur_data = self.data[idx]
        if len(cur_data) != len(self.transforms):
            raise ValueError("data must have the same number of columns "
                             "with transforms function")
        return [self.transforms[i](cur_data[i]) for i in range(len(self.transforms))]

    def __len__(self):
        return len(self.data)

    def get_vocabs(self):
        return self.vocabs
