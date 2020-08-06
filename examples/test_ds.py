import os
from typing import Dict, List, Union, Iterable
import sys
from collections import Counter, defaultdict

import tensorflow as tf

sys.path.append("../")
import arcnlp.tf


train_path = os.path.expanduser("~/datasets/LCQMC/train_seg.txt")
print(train_path)


def tokenizer(text):
    import jieba
    return jieba.lcut(text)


builder = arcnlp.tf.data.TextMatchingData(
    arcnlp.tf.data.TextFeature(tokenizer),
    arcnlp.tf.data.Label())
# train_ds = builder.raw_dataset(train_path)
train_examples = list(builder.read_from_path(train_path))
for ex in train_examples[:3]:
    print(ex)
builder.build_vocab(train_examples)
train_ds = builder.build_dataset(train_path)
for x in train_ds.take(3):
    print(x)

for batch in builder.get_batches(train_ds).take(3):
    print(batch)

print(len(builder.text_feature.vocab))

text_embedder = tf.keras.layers.Embedding(len(builder.text_feature.vocab),
                                          200, mask_zero=True)

model = arcnlp.tf.models.BiLstmMatching(builder.features,
                                        builder.targets,
                                        text_embedder)
model.summary()
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=['acc'])
model.fit(builder.get_bucket_batches(train_ds), epochs=3)
