# -*- coding: utf-8 -*-

from tensorflow import keras

from arcnlp import keras_ext

word_field = keras_ext.data.Field(pad_token='<pad>', init_token="<s>",
                                  eos_token="</s>", unk_token='<unk>')
text_fields = {"word": word_field}

label_field = keras_ext.data.LabelField()
label_fields = {'label': label_field}

imdb = keras_ext.datasets.IMDB("/Users/linhx13/datasets/aclImdb/test",
                               text_fields=text_fields,
                               label_field=label_field)

print(len(imdb))

word_field.build_vocab(imdb)
label_field.build_vocab(imdb)

print(len(word_field.vocab))
print(len(label_field.vocab))

batch = keras_ext.data.Batch(imdb.examples[:16], imdb)

x, y = batch.as_tensors()
print(x)
print(y)

print(imdb.examples[0].__dict__)

data_iter = keras_ext.data.BucketIterator(imdb, batch_size=16)

# for batch in data_iter:
#     # print(batch.word.shape, batch.label.shape)
#     pass

# for batch in data_iter:
#     print(batch.word.shape, batch.label.shape)


text_field_embedder = \
    keras_ext.layers.text_field_embedders.BasicTextFieldEmbedder({
        'word': keras.layers.Embedding(len(word_field.vocab), 300,
                                       mask_zero=True)
    })
model = keras_ext.models.TextClassifier(
    text_fields, label_field,
    text_field_embedder, seq2vec_encoder=keras.layers.LSTM(100))
model.build_model().summary()
