#!/bin/bash

rm -f sentiment_classification-dataset-1.0.0.tar.gz
wget --no-check-certificate https://baidu-nlp.bj.bcebos.com/sentiment_classification-dataset-1.0.0.tar.gz
tar -zxvf sentiment_classification-dataset-1.0.0.tar.gz

python ./convert_to_fasttext.py \
       --input_path ./senta_data/train.tsv \
       --output_path ./senta_data/train_fasttext.txt \
       --text_column text_a \
       --label_column label

python ./convert_to_fasttext.py \
       --input_path ./senta_data/test.tsv \
       --output_path ./senta_data/test_fasttext.txt \
       --text_column text_a \
       --label_column label

python ./convert_to_fasttext.py \
       --input_path ./senta_data/dev.tsv \
       --output_path ./senta_data/dev_fasttext.txt \
       --text_column text_a \
       --label_column label

rm -f sentiment_classification-dataset-1.0.0.tar.gz