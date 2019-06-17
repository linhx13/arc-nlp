# -*- coding: utf-8 -*-

from arcnlp.keras_ext import data

data_list = [
    ["Hello, world!"],
    ["Good morning"],
    ["How are you?"]
]

word_field = data.Field(pad_token='<pad>', init_token="<s>", eos_token="</s>",
                        unk_token='<unk>', fix_length=7)

nesting_field = data.Field(pad_token="<c>", init_token="<w>", eos_token="</w>")
char_field = data.NestedField(nesting_field, init_token="<s>",
                              eos_token="</s>")

# fields = [('word', word_field)]
fields = [[('word', word_field), ('char', char_field)]]

dataset = data.TabularDataset(data_list, "list", fields)

word_field.build_vocab(dataset)

print(word_field.vocab.unk_token, word_field.vocab.unk_index)
print(word_field.vocab.stoi)
print(word_field.vocab.itos)

print(word_field.pad(dataset.word))
print(word_field.process(dataset.word))

print('=' * 20)

char_field.build_vocab(dataset)
print(char_field.vocab.unk_token, char_field.vocab.unk_index)
print(char_field.vocab.stoi)
print(char_field.vocab.itos)
