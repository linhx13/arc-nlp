# -*- coding: utf-8 -*-

import os

import tensorflow as tf

import arcnlp.tf
from arcnlp.tf.losses import crf_loss
from arcnlp.tf.metrics import crf_accuracy

tf.compat.v1.disable_v2_behavior()

train_path = os.path.expanduser("~/datasets/conll-corpora/CoNLL-2003/eng.train")
test_path = os.path.expanduser("~/datasets/conll-corpora/CoNLL-2003/eng.testa")

model_dir = './conll2003_dir'
arcnlp.tf.utils.mkdir_p(model_dir)

token_fields = {
    'word': arcnlp.tf.data.Field()
}

data_handler = arcnlp.tf.data.Conll2003DataHandler(
    token_fields=token_fields,
    tag_column="ner",
    feature_columns=['pos', 'chunk'],
    sort_feature="tokens.word")

train_dataset = data_handler.create_dataset_from_path(train_path)
test_dataset = data_handler.create_dataset_from_path(test_path)

print('train_dataset size: %d' % len(train_dataset))
print('test_dataset size: %d' % len(test_dataset))

# data = next(iter(train_dataset))
data = train_dataset[0]
for n, f in train_dataset.fields.items():
    print(n, f, f.is_target)
    print(getattr(data, n))

# for n, f in train_dataset.fields.items():
#     f.build_vocab(train_dataset, test_dataset)
#     print("build_vocab for %s done, vocab size: %d" % (n, len(f.vocab)))

data_handler.build_vocab(train_dataset, test_dataset)

print(train_dataset.fields['tags'].vocab.itos)
print(data_handler.fields['tags'].vocab.itos)

batch = arcnlp.tf.data.Batch(train_dataset.examples[:1], train_dataset)
x = dict((f, getattr(batch, f)) for f in data_handler.features)
y = dict((f, getattr(batch, f)) for f in data_handler.targets)
print(x)
print(y)
print(len(x))
print(len(y))


text_embedder = arcnlp.tf.layers.text_embedders.BasicTextEmbedder(
    {'word': tf.keras.layers.Embedding(len(token_fields['word'].vocab), 300)}
)

feature_embedders = {
    'pos': tf.keras.layers.Embedding(
        len(data_handler.fields['pos'].vocab), 50),
    'chunk': tf.keras.layers.Embedding(
        len(data_handler.fields['chunk'].vocab), 50)
}

seq2seq_encoder = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(100, return_sequences=True))

arcnlp.tf.utils.config_tf_gpu()

tagger = arcnlp.tf.models.CrfTagger(
    features=data_handler.features,
    targets=data_handler.targets,
    text_embedder=text_embedder,
    encoder=seq2seq_encoder,
    feature_embedders=feature_embedders)

trainer = arcnlp.tf.training.Trainer(tagger, data_handler,
                                     optimizer='adam',
                                     loss=crf_loss,
                                     metrics=[crf_accuracy])
trainer.train(train_dataset=train_dataset,
              validation_dataset=test_dataset,
              validation_metric='val_crf_accuracy',
              batch_size=32,
              epochs=3,
              model_dir=model_dir)

eval_res = trainer.evaluate(test_dataset)
print(eval_res)

new_trainer = arcnlp.tf.training.Trainer.from_path('./conll2003_dir')
print(new_trainer.model.name)
eval_res = new_trainer.evaluate(test_dataset)
print(eval_res)

