import json
import os
from collections import Counter
import time
from functools import partial

import torch
from torch.nn import Conv1d, Linear
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchtext.experimental.datasets.text_classification import TextClassificationDataset
from torchtext.experimental.functional import sequential_transforms, vocab_func, totensor
from torchtext.vocab import Vocab, build_vocab_from_iterator
# from allennlp.modules.seq2vec_encoders import CnnEncoder
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
    # examples = examples[:2048]

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


class CnnEncoder(torch.nn.Module):

    def __init__(self,
                 input_dim,
                 num_filters,
                 kernel_sizes=(2,3,4,5),
                 conv_layer_activation=None,
                 output_dim:int=None):
        super(CnnEncoder, self).__init__()
        self.input_dim = input_dim
        self.conv_layers = torch.nn.ModuleList([
            Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size)
            for kernel_size in kernel_sizes])
        self.conv_layer_activation = conv_layer_activation or F.relu
        maxpool_output_dim = num_filters  * len(kernel_sizes)
        if output_dim:
            self.projection_layer = Linear(maxpool_output_dim, output_dim)
            self.output_dim = output_dim
        else:
            self.projection_layer = None
            self.output_dim = maxpool_output_dim

    def get_input_dim(self) -> int:
        return self.input_dim

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self, tokens: torch.Tensor, mask: torch.BoolTensor=None):
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1)

        tokens = torch.transpose(tokens, 1, 2)
        # print('tokens:', tokens)
        # conv_outputs = [conv_layer(tokens) for conv_layer in self.conv_layers]
        # print('conv_outputs:', conv_outputs)
        # act_outputs = [self.conv_layer_activation(x) for x in conv_outputs]
        # print('act_outputs:', act_outputs)
        # filter_outputs = [x.max(dim=2)[0] for x in act_outputs]
        filter_outputs = [self.conv_layer_activation(conv_layer(tokens)).max(dim=2)[0]
                          for conv_layer in self.conv_layers]
        # print('filter_outputs:', filter_outputs)
        maxpool_output = torch.cat(filter_outputs, dim=1) \
            if len(filter_outputs) > 1 else filter_outputs[0]
        # print('maxpool_output:', maxpool_output)
        if self.projection_layer:
            res = self.projection_layer(maxpool_output)
        else:
            res = maxpool_output
        # print('res:', res)
        return res


def get_tokens_mask(tokens, padding_idx=0) -> torch.BoolTensor:
    return tokens != padding_idx


class Model(torch.nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class, padding_idx, dropout=0.5):
        super(Model, self).__init__()
        # self.embedding = torch.nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.encoder = CnnEncoder(embed_dim, 128)
        self.fc = torch.nn.Linear(self.encoder.get_output_dim(), num_class)
        self.dropout = torch.nn.Dropout(dropout)
        # self.init_weights()

    def init_weights(self):
        init_range = 0.5
        # self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()

    def forward(self, tokens):
        if self.embedding.padding_idx is not None:
            mask = get_tokens_mask(tokens, self.embedding.padding_idx)
        else:
            mask = None
        embedded = self.embedding(tokens)
        # print('embedded:', embedded)
        encoded = self.encoder(embedded, mask)
        encoded = self.dropout(encoded)
        # print('encoded:', encoded)
        return self.fc(encoded)


# VOCAB_SIZE = len(train_dataset.get_vocab())
VOCAB_SIZE = len(text_vocab)
EMBED_DIM = 300
NUM_CLASS = len(label_vocab)
# model = Model(VOCAB_SIZE, EMBED_DIM, NUM_CLASS).to(device)
model = Model(len(text_vocab), EMBED_DIM, NUM_CLASS, padding_idx=text_vocab['<pad>']).to(device)
print(model)


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
    # for texts, offsets, labels in train_data:
    for texts, labels in train_data:
        # print('texts:', texts)
        # print('labels:', labels)
        optimizer.zero_grad()
        # print(texts)
        # print(offsets)
        # print(labels)
        # texts, offsets, labels = texts.to(device), offsets.to(device), labels.to(device)
        # preds = model(texts, offsets)
        texts, labels = texts.to(device), labels.to(device)
        preds = model(texts)
        # print('preds:', preds)
        # print("labels:", labels)
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
    # for texts, offsets, labels in test_data:
    for texts, labels in test_data:
        # texts, offsets, labels = texts.to(device), offsets.to(device), labels.to(device)
        texts, labels = texts.to(device), labels.to(device)
        with torch.no_grad():
            # preds = model(texts, offsets)
            preds = model(texts)
            loss = criterion(preds, labels)
            loss += loss.item()
            acc += (preds.argmax(1) == labels).sum().item()

    return loss / len(test_data.dataset), acc / len(test_data.dataset)

# print(list(model.parameters()))
criterion = torch.nn.CrossEntropyLoss().to(device)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0)
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
# scheduler = None

N_EPOCHS = 20
BATCH_SIZE = 32
min_valid_loss = float('inf')

train_data = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=partial(collate_fn, padding_idx=text_vocab['<pad>']),
                        num_workers=4)
valid_data = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=partial(collate_fn, padding_idx=text_vocab['<pad>']),
                        num_workers=4)

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
