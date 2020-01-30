# -*- coding: utf-8 -*-

import os
import pickle

from sklearn.model_selection import train_test_split
import tensorflow as tf

import arcnlp.tf

train_path = '/opt/userhome/ichongxiang/datasets/quora-question-pairs/train.csv'
test_path = '/opt/userhome/ichongxiang/datasets/quora-question-pairs/test.csv'

tf.compat.v1.disable_eager_execution()

model_dir = "quora_question_pairs_model"
arcnlp.tf.utils.mkdir_p(model_dir)

word_field = arcnlp.tf.data.Field()
token_fields = {
    'word': word_field
}

tokenizer = arcnlp.tf.data.tokenizers.SpacyTokenizer()

data_handler = arcnlp.tf.data.QuoraQuestionPairsDataHandler(
    token_fields, sparse_target=False)

train_dataset = data_handler.create_dataset_from_path(train_path)
train_examples, dev_examples = train_test_split(train_dataset.examples,
                                                test_size=0.1)
print(train_examples[:5])
print(dev_examples[:5])
train_dataset = arcnlp.tf.data.Dataset(train_examples, train_dataset.fields)
dev_dataset = arcnlp.tf.data.Dataset(dev_examples, train_dataset.fields)
# test_dataset = data_handler.create_dataset_from_path(test_path)

print(train_dataset.fields)
print(dev_dataset.fields)

data = train_dataset[0]
for n, f in train_dataset.fields.items():
    print(n, f)
    print(getattr(data, n))

data_handler.build_vocab(train_dataset, dev_dataset)
for n, f in data_handler.features.items():
    print(n, len(f.vocab))
for n, f in data_handler.targets.items():
    print(n, len(f.vocab))


with open(os.path.join(model_dir, "data_handler.pkl"), "wb") as fout:
    pickle.dump(data_handler, fout)

token_embedders = {
    'word': tf.keras.layers.Embedding(len(word_field.vocab), 300, mask_zero=True)
}
text_embedder = arcnlp.tf.layers.text_embedders.BasicTextEmbedder(
    token_embedders)

arcnlp.tf.utils.config_tf_gpu()

# model_cls = arcnlp.tf.models.ESIM
model_cls = arcnlp.tf.models.BiMPM

model = model_cls(data_handler.features,
                  data_handler.targets,
                  text_embedder)

trainer = arcnlp.tf.training.Trainer(model, data_handler,
                                     optimizer="adam",
                                     loss='categorical_crossentropy',
                                     metrics=['acc'])
trainer.train(train_dataset=train_dataset,
              validation_dataset=dev_dataset,
              batch_size=32,
              epochs=3,
              model_dir=model_dir)

eval_res = trainer.evaluate(dev_dataset)
print(eval_res)

new_trainer = arcnlp.tf.training.Trainer.from_path(model_dir)
eval_res = trainer.evaluate(dev_dataset)
print(eval_res)
