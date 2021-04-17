import os
from collections import Counter
import time
from functools import partial

import jieba
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchtext.experimental.datasets.text_classification import (
    TextClassificationDataset,
)
from torchtext.experimental.functional import (
    sequential_transforms,
    vocab_func,
    totensor,
)
from torchtext.vocab import Vocab, build_vocab_from_iterator, Vectors

from arcnlp.torch.models import *
from arcnlp.torch.nn import BOWEncoder

jieba.initialize()

DATA_DIR = "/mnt/8f00be1b-84a6-4b38-8b59-47075910175b/datasets/THUCNews_small"
train_path = os.path.expanduser(os.path.join(DATA_DIR, "train.txt"))
valid_path = os.path.expanduser(os.path.join(DATA_DIR, "dev.txt"))
class_path = os.path.expanduser(os.path.join(DATA_DIR, "class.txt"))

embedding_path = "/mnt/8f00be1b-84a6-4b38-8b59-47075910175b/datasets/sgns.sogounews.bigram-char"


def tokenizer(text):
    return jieba.lcut(text)


def build_dataset(data_path, vocab):
    examples = []
    with open(data_path) as fin:
        for line in fin:
            text, label = line.strip("\r\n").split("\t")
            examples.append((label, text))
    # examples = examples[:2048]

    text_transform = sequential_transforms(tokenizer)
    if vocab is None:
        vocab = build_vocab_from_iterator(
            text_transform(text) for _, text in examples
        )

    label_transform = sequential_transforms(int)
    text_transform = sequential_transforms(text_transform, vocab_func(vocab))
    dataset = TextClassificationDataset(
        examples, vocab, (label_transform, text_transform)
    )
    return dataset


train_dataset = build_dataset(train_path, None)
vocab = train_dataset.get_vocab()
valid_dataset = build_dataset(valid_path, vocab)

with open(class_path) as fin:
    class_list = [line.strip() for line in fin]

print(len(train_dataset))
print(len(valid_dataset))
print(len(vocab), len(class_list))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
padding_idx = vocab["<pad>"]
print("padding_idx", padding_idx)


vectors = Vectors(embedding_path)
vocab.load_vectors(vectors)

# embedding = nn.Embedding(len(vocab), 300, padding_idx=padding_idx)
embedding = nn.Embedding.from_pretrained(
    vocab.vectors, freeze=False, padding_idx=padding_idx
)


def build_model(arch, num_classes, embedding, padding_idx):
    if arch == "BOW":
        return TextClassifier(
            num_classes,
            embedding,
            seq2vec_encoder=BOWEncoder(embedding.embedding_dim),
            padding_idx=padding_idx,
        )
    elif arch == "TextCNN":
        return TextCNN(num_classes, embedding, padding_idx=padding_idx)
    elif arch == "TextRNN":
        return TextRNN(num_classes, embedding, padding_idx=padding_idx)
    elif arch == "TextRCNN":
        return TextRCNN(num_classes, embedding, padding_idx=padding_idx)


# arch = "BOW"
# arch = "TextCNN"
arch = "TextRNN"
model = build_model(arch, len(class_list), embedding, padding_idx)


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
            padded_texts.append(text + [padding_idx] * (max_len - len(text)))
    padded_texts = torch.LongTensor(padded_texts)
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
        loss.backward()
        optimizer.step()
        train_acc += (preds.argmax(1) == labels).sum().item()

    if scheduler is not None:
        scheduler.step()
    return train_loss / len(train_data.dataset), train_acc / len(
        train_data.dataset
    )


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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)


N_EPOCHS = 10
BATCH_SIZE = 128
min_valid_loss = float("inf")

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=partial(collate_fn, padding_idx=padding_idx),
    num_workers=4,
)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=partial(collate_fn, padding_idx=padding_idx),
    num_workers=4,
)

model = model.to(device)
for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train_func(
        train_dataloader, model, criterion, optimizer, scheduler, device
    )
    valid_loss, valid_acc = test_func(
        valid_dataloader, model, criterion, device
    )

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print(
        "Epoch: %d" % (epoch + 1),
        " | time in %d minutes, %d seconds" % (mins, secs),
    )
    print(
        f"\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)"
    )
    print(
        f"\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)"
    )


print("Checking the results of valid dataset...")
valid_loss, valid_acc = test_func(valid_dataloader, model, criterion, device)
print(f"\tLoss: {valid_loss:.4f}(test)\t|\tAcc: {valid_acc * 100:.1f}%(test)")
