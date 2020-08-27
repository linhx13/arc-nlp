import json
import os
from collections import Counter
import time
from functools import partial

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchtext.experimental.datasets.text_classification import TextClassificationDataset
from torchtext.experimental.functional import sequential_transforms, vocab_func, totensor
from torchtext.vocab import Vocab, build_vocab_from_iterator
from arcnlp.torch.models import TextCNN, TextRNN
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
    examples = examples[:2048]

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


embedding = nn.Embedding(len(text_vocab), 300, padding_idx=text_vocab['<pad>'])

# model_cls = TextCNN
model_cls = TextRNN
model = model_cls(len(label_vocab), embedding)


def collate_fn(batch, padding_idx, max_len=None):
    labels = torch.LongTensor([entry[0] for entry in batch])
    texts = [entry[1] for entry in batch]
    if max_len is None:
        max_len = max(len(text) for text in texts)
    padded_texts = []
    for text in texts:
        if len(text) > max_len:
            padded_texts.append(text[:max_len])
        else:
            padded_texts.append(text+ [padding_idx] * (max_len - len(text)))
    # padded_texts = [torch.LongTensor(text) for text in padded_texts]
    padded_texts = torch.LongTensor(padded_texts)
    # offsets = [0] + [len(entry) for entry in texts]
    # offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    # padded_texts = torch.cat(padded_texts)
    # return texts, offsets, labels
    return padded_texts, labels


def train_func(train_data, model, criterion, optimizer, scheduler, device):
    model.train()
    train_loss = 0
    train_acc = 0
    for texts, labels in train_data:
        optimizer.zero_grad()
        texts, labels = texts.to(device), labels.to(device)
        preds = model(texts)
        loss = criterion(preds, labels)
        train_loss += loss.item()
        print(loss.item())
        loss.backward()
        optimizer.step()
        train_acc += (preds.argmax(1) == labels).sum().item()

    if scheduler is not None:
        scheduler.step()
    return train_loss / len(train_data.dataset), train_acc / len(train_data.dataset)


def test_func(test_data, model, criterion, device):
    model.eval()
    loss = 0
    acc = 0
    for texts, labels in test_data:
        texts, labels = texts.to(device), labels.to(device)
        with torch.no_grad():
            preds = model(texts)
            loss = criterion(preds, labels)
            loss += loss.item()
            acc += (preds.argmax(1) == labels).sum().item()

    return loss / len(test_data.dataset), acc / len(test_data.dataset)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

N_EPOCHS = 20
BATCH_SIZE = 32
min_valid_loss = float('inf')

train_data = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=partial(collate_fn, padding_idx=text_vocab['<pad>']),
                        num_workers=4)
valid_data = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=partial(collate_fn, padding_idx=text_vocab['<pad>']),
                        num_workers=4)

model = model.to(device)
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
