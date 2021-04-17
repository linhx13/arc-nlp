import os
from collections import Counter
import time
from functools import partial

import jieba
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
    ngrams_func,
)
from torchtext.vocab import Vocab, build_vocab_from_iterator, Vectors
from torchtext.data.utils import ngrams_iterator

from arcnlp.torch.models import *

import pytorch_lightning as pl

jieba.initialize()

DATA_DIR = "/mnt/8f00be1b-84a6-4b38-8b59-47075910175b/datasets/THUCNews_small"
train_path = os.path.expanduser(os.path.join(DATA_DIR, "train.txt"))
valid_path = os.path.expanduser(os.path.join(DATA_DIR, "dev.txt"))
class_path = os.path.expanduser(os.path.join(DATA_DIR, "class.txt"))

EMBEDDING_PATH = "/mnt/8f00be1b-84a6-4b38-8b59-47075910175b/datasets/sgns.sogounews.bigram-char"

MAX_EPOCHS = 10
BATCH_SIZE = 128
FASTTEXT_NGRAMS_BUCKETS = 100000

ARCH = "FastText"


def build_model(arch, num_classes, vocab, embedding_path=None):
    padding_idx = vocab["<pad>"]
    if embedding_path is not None:
        vectors = Vectors(embedding_path)
        vocab.load_vectors(vectors)
        embedding = nn.Embedding.from_pretrained(
            vocab.vectors, freeze=False, padding_idx=padding_idx
        )
    else:
        embedding = nn.Embedding(len(vocab), 300, padding_idx=padding_idx)
    if arch == "BOW":
        from arcnlp.torch.nn import BOWEncoder

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
    elif arch == "FastText":
        ngrams_embedding = nn.Embedding(
            len(vocab) + FASTTEXT_NGRAMS_BUCKETS, 300, padding_idx=padding_idx
        )
        ngrams_embedding.weight.data[: len(vocab)] = embedding.weight.data
        return FastText(num_classes, ngrams_embedding, padding_idx=padding_idx)


def tokenizer(text):
    return jieba.lcut(text)


def ngrams_hash(buckets, base=0):
    def bigram_hash(ids, t):
        x1 = ids[t - 1] if t - 1 >= 0 else 0
        return (x1 * 14918087 + ids[t]) % buckets + base

    def trigram_hash(ids, t):
        x1 = ids[t - 1] if t - 1 >= 0 else 0
        x2 = ids[t - 2] if t - 2 >= 0 else 0
        return (
            x2 * 14918087 * 18408749 + x1 * 14918087 + ids[t]
        ) % buckets + base

    def func(ids):
        ngrams = ids[:]
        for t in range(len(ids)):
            ngrams.append(bigram_hash(ids, t))
            ngrams.append(trigram_hash(ids, t))
        return ngrams

    return func


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
    if ARCH == "FastText":
        text_transform = sequential_transforms(
            text_transform,
            ngrams_hash(buckets=FASTTEXT_NGRAMS_BUCKETS, base=len(vocab)),
        )

    dataset = TextClassificationDataset(
        examples, vocab, (label_transform, text_transform)
    )
    return dataset


def load_data(train_path, valid_path, class_path):
    train_dataset = build_dataset(train_path, vocab=None)
    valid_dataset = build_dataset(valid_path, vocab=train_dataset.get_vocab())
    with open(class_path) as fin:
        class_list = [line.strip() for line in fin]
    return train_dataset, valid_dataset, class_list


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


def build_dataloader(dataset, batch_size, shuffle, padding_idx):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_fn, padding_idx=padding_idx),
        num_workers=4,
    )


class Task(pl.LightningModule):
    def __init__(self, model):
        super(Task, self).__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.train_accuracy = pl.metrics.Accuracy()
        self.valid_accuracy = pl.metrics.Accuracy()

    def forward(self, tokens):
        return self.model(tokens)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def step(self, batch, batch_idx):
        tokens, labels = batch
        outputs = self(tokens)
        loss = self.criterion(outputs, labels.long())
        preds = torch.argmax(outputs, axis=1)
        return {"loss": loss, "preds": preds}

    def training_step(self, batch, batch_idx):
        res = self.step(batch, batch_idx)
        acc = self.train_accuracy(res["preds"], batch[1])
        self.log(
            "train_loss",
            res["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )
        self.log(
            "train_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )
        return res["loss"]

    def validation_step(self, batch, batch_idx):
        res = self.step(batch, batch_idx)
        acc = self.valid_accuracy(res["preds"], batch[1])
        self.log(
            "val_loss",
            res["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )
        self.log(
            "val_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )


def run():
    train_dataset, valid_dataset, class_list = load_data(
        train_path, valid_path, class_path
    )
    vocab = train_dataset.get_vocab()
    train_dataloader = build_dataloader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        padding_idx=vocab["<pad>"],
    )
    valid_dataloader = build_dataloader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        padding_idx=vocab["<pad>"],
    )
    model = build_model(ARCH, len(class_list), vocab, EMBEDDING_PATH)
    task = Task(model)
    early_stopping = pl.callbacks.EarlyStopping("val_loss", patience=3)
    trainer = pl.Trainer(
        gpus=-1 if torch.cuda.is_available() else 0,
        max_epochs=MAX_EPOCHS,
        callbacks=[early_stopping],
        logger=False,
        checkpoint_callback=False,
    )
    trainer.fit(task, train_dataloader, valid_dataloader)


if __name__ == "__main__":
    pl.seed_everything(42)
    run()
