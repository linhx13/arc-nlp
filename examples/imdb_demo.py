# -*- coding: utf-8 -*-

import os

import tensorflow as tf

import arcnlp.tf

tf.compat.v1.disable_eager_execution()

train_path = os.path.expanduser("~/datasets/aclImdb/train")
test_path = os.path.expanduser("~/datasets/aclImdb/test")

model_dir = './imdb_dir'

arcnlp.tf.utils.mkdir_p(model_dir)

word_field = arcnlp.tf.data.Field()

token_fields = {
    'word': word_field
}

data_handler = arcnlp.tf.data.IMDBDataHandler(token_fields)

train_dataset = data_handler.create_dataset_from_path(train_path)
test_dataset = data_handler.create_dataset_from_path(test_path)

print('train_dataset size: %d' % len(train_dataset))
print('test_dataset size: %d' % len(test_dataset))

print(train_dataset.fields)

data = train_dataset[0]
for n, f in train_dataset.fields.items():
    print(n, f, f.is_target)
    print(getattr(data, n))

data_handler.build_vocab(train_dataset, test_dataset)
for n, f in data_handler.features.items():
    print(n, len(f.vocab))
for n, f in data_handler.targets.items():
    print(n, len(f.vocab))

batch = arcnlp.tf.data.Batch(train_dataset.examples[:1], train_dataset)
x, y = batch
print(x)
print(y)
print(len(x))
print(len(y))

text_embedder = arcnlp.tf.layers.text_embedders.BasicTextEmbedder({
    'word': tf.keras.layers.Embedding(len(word_field.vocab), 300, mask_zero=True)
})

# seq2vec_encoder = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(100))
# seq2vec_encoder = arcnlp_tf.layers.seq2vec_encoders.BOWEncoder()
seq2vec_encoder = arcnlp.tf.layers.seq2vec_encoders.CNNEncoder()

arcnlp.tf.utils.config_tf_gpu()

text_cat = arcnlp.tf.models.TextCNNClassifier(
    data_handler.features,
    data_handler.targets,
    text_embedder)

trainer = arcnlp.tf.training.Trainer(text_cat, data_handler)
trainer.train(train_dataset=train_dataset,
              validation_dataset=test_dataset,
              batch_size=32,
              epochs=3,
              model_dir=model_dir)

eval_res = trainer.evaluate(test_dataset)
print(eval_res)

new_trainer = arcnlp.tf.training.Trainer.from_path(model_dir)
print(new_trainer.model.name)
eval_res = new_trainer.evaluate(test_dataset)
print(eval_res)
