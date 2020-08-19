import json
import os
from collections import Counter
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchtext.experimental.datasets.text_classification import TextClassificationDataset
from torchtext.experimental.functional import sequential_transforms, vocab_func, totensor
from torchtext.vocab import Vocab, build_vocab_from_iterator
import jieba
import numpy as np

jieba.initialize()

label_path = os.path.expanduser("~/datasets/tnews_public/labels.json")
train_path = os.path.expanduser("~/datasets/tnews_public/train.json")
valid_path = os.path.expanduser("~/datasets/tnews_public/dev.json")


def tokenizer(text):
    return jieba.lcut(text)


def label_func(vocab, sparse=True):
    def func(x):
        if isinstance(x, list):
            x = [vocab[t] for t in x]
        else:
            x = vocab[x]
        if sparse:
            return x
        else:
            return np.eye(len(vocab), dtype='int32')[x]

    return func


def build_dataset(data_path, label_path, text_vocab=None, label_vocab=None):
    examples = []
    with open(data_path) as fin:
        for line in fin:
            data = json.loads(line)
            example = ('%s:%s' % (data['label'], data['label_desc']), data['sentence'])
            examples.append(example)

    text_transform = sequential_transforms(tokenizer)
    if text_vocab is None:
        text_vocab = build_vocab_from_iterator(text_transform(ex[1]) for ex in examples)

    if label_vocab is None:
        label_counter = Counter()
        with open(label_path) as fin:
            for line in fin:
                data = json.loads(line)
                label_counter['%s:%s' % (data['label'], data['label_desc'])] += 1
        label_vocab = Vocab(label_counter, specials=[])

    label_transform= sequential_transforms(label_func(label_vocab))
    text_transform = sequential_transforms(text_transform, vocab_func(text_vocab))
    dataset = TextClassificationDataset(examples, (label_vocab, text_vocab),
                                        (label_transform, text_transform))
    return dataset, text_vocab, label_vocab


train_dataset, text_vocab, label_vocab = build_dataset(train_path, label_path)
valid_dataset, _, _ = build_dataset(valid_path, label_path, text_vocab=text_vocab, label_vocab=label_vocab)
print(len(train_dataset))
print(len(valid_dataset))
print(label_vocab.stoi)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(torch.nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(Model, self).__init__()
        self.embedding = torch.nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = torch.nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        init_range = 0.5
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


# VOCAB_SIZE = len(train_dataset.get_vocab())
VOCAB_SIZE = len(text_vocab)
EMBED_DIM = 300
NUM_CLASS = len(label_vocab)
model = Model(VOCAB_SIZE, EMBED_DIM, NUM_CLASS).to(device)
print(model)


def collate_fn(batch):
    labels = torch.tensor([entry[0] for entry in batch])
    texts = [torch.tensor(entry[1]) for entry in batch]
    offsets = [0] + [len(entry) for entry in texts]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    texts = torch.cat(texts)
    return texts, offsets, labels


def train_func(train_data, model, criterion, optimizer, scheduler, device):
    train_loss = 0
    train_acc = 0
    for texts, offsets, labels in train_data:
        optimizer.zero_grad()
        # print(texts)
        # print(offsets)
        # print(labels)
        texts, offsets, labels = texts.to(device), offsets.to(device), labels.to(device)
        preds = model(texts, offsets)
        loss = criterion(preds, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (preds.argmax(1) == labels).sum().item()

    scheduler.step()
    return train_loss / len(train_data.dataset), train_acc / len(train_data.dataset)


def test_func(test_data, model, criterion, device):
    loss = 0
    acc = 0
    for texts, offsets, labels in test_data:
        texts, offsets, labels = texts.to(device), offsets.to(device), labels.to(device)
        with torch.no_grad():
            preds = model(texts, offsets)
            loss = criterion(preds, labels)
            loss += loss.item()
            acc += (preds.argmax(1) == labels).sum().item()

    return loss / len(test_data.dataset), acc / len(test_data.dataset)


criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

N_EPOCHS = 5
BATCH_SIZE = 8
min_valid_loss = float('inf')

train_data = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate_fn)
valid_data = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                      collate_fn =collate_fn)

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train_func(train_data, model, criterion, optimizer,
                                       scheduler, device)
    valid_loss, valid_acc = test_func(valid_data, model, criterion, device)

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')


print('Checking the results of valid dataset...')
valid_loss, valid_acc = test_func(valid_data, model, criterion, device)
print(f'\tLoss: {valid_loss:.4f}(test)\t|\tAcc: {valid_acc * 100:.1f}%(test)')
