# -*- coding: utf-8 -*-

import logging

import tensorflow as tf

from sklearn.model_selection import train_test_split

import arcnlp.tf
from arcnlp.tf.losses import crf_loss
from arcnlp.tf.metrics import crf_accuracy

logger = logging.getLogger(__name__)


def create_datasets(args, data_handler):
    if not args.test_path and not args.test_size:
        raise ValueError("test_path and test_size cannot both be none")
    if args.test_path:
        train_dataset = data_handler.create_dataset_from_path(args.train_path)
        test_dataset = data_handler.create_dataset_from_path(args.test_path)
    else:
        dataset = data_handler.create_dataset_from_path(args.train_path)
        train_examples, test_examples = train_test_split(
            dataset.examples, test_size=args.test_size)
        train_dataset = arcnlp.tf.data.Dataset(train_examples, dataset.fields)
        test_dataset = arcnlp.tf.data.Dataset(test_examples, dataset.fields)
    return train_dataset, test_dataset


def run_train(args):
    arcnlp.tf.utils.mkdir_p(args.model_dir)
    token_fields = {
        'char': arcnlp.tf.data.Field()
    }
    data_handler = arcnlp.tf.data.NerDataHanlder(
        token_fields, use_seg_feature=args.use_seg_feature)
    train_dataset, test_dataset = create_datasets(args, data_handler)
    logger.info("train examples: %d, test examples: %d" %
                (len(train_dataset), len(test_dataset)))
    data_handler.build_vocab(train_dataset, test_dataset)
    logger.info("data handler build vocab done")
    text_embedder = arcnlp.tf.layers.text_embedders.BasicTextEmbedder({
        'char': tf.keras.layers.Embedding(len(token_fields['char'].vocab), 300)
    })
    if args.use_seg_feature:
        feature_embedders = {
            'seg': tf.keras.layers.Embedding(len(data_handler.fields['seg'].vocab), 30)
        }
    else:
        feature_embedders = {}
    if args.use_cudnn:
        seq2seq_encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.CuDNNLSTM(100, return_sequences=True))
    else:
        seq2seq_encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(100, return_sequences=True))
    tagger = arcnlp.tf.models.CrfTagger(
        features=data_handler.features,
        targets=data_handler.targets,
        text_embedder=text_embedder,
        feature_embedders=feature_embedders,
        encoder=seq2seq_encoder)
    arcnlp.tf.utils.config_tf_gpu()

    trainer = arcnlp.tf.training.Trainer(tagger, data_handler,
                                         optimizer='adam',
                                         loss=crf_loss,
                                         metrics=[crf_accuracy])
    trainer.train(train_dataset=train_dataset,
                  validation_dataset=test_dataset,
                  validation_metric='val_crf_accuracy',
                  batch_size=args.batch_size,
                  epochs=args.epochs,
                  model_dir=args.model_dir)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s %(asctime)s %(filename)s:%(lineno)d %(message)s')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--test_path")
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--use_seg_feature",
                        default=False, action='store_true')
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--use_cudnn", default=False, action='store_true',
                        help="Use CuDNN based LSTM cells.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    logger.info('args: %s' % args)

    run_train(args)
