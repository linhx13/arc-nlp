import os

import tensorflow as tf
import arcnlp.tf

# tf.compat.v1.disable_eager_execution()
arcnlp.tf.utils.config_tf_gpu()


train_path = os.path.expanduser("~/datasets/LCQMC/train_seg.txt")
val_path = os.path.expanduser("~/datasets/LCQMC/test_seg.txt")
print(train_path, val_path)


def tokenizer(text):
    import jieba
    return jieba.lcut(text)


builder = arcnlp.tf.data.TextMatchingData(
    arcnlp.tf.data.TextFeature(tokenizer),
    arcnlp.tf.data.Label())
train_examples = list(builder.read_from_path(train_path))
builder.build_vocab(train_examples)
print(len(builder.text_feature.vocab))
print(len(builder.label.vocab))
train_dataset = builder.build_dataset(train_path)
print('train_dataset', train_dataset)
val_dataset = builder.build_dataset(val_path)
print('val_dataset', val_dataset)

text_embedder = tf.keras.layers.Embedding(len(builder.text_feature.vocab),
                                          200, mask_zero=True)

model = arcnlp.tf.models.BiLstmMatching(builder.features,
                                        builder.targets,
                                        text_embedder)
model.summary()

trainer = arcnlp.tf.training.Trainer(model, builder)
trainer.train(train_dataset=train_dataset,
              val_dataset=val_dataset,
              epochs=3,
              model_dir='./bilstm_matching_model',)
