# -*- coding: utf-8 -*-

import logging
import os

import tensorflow as tf
import arcnlp.tf

logger = logging.getLogger(__name__)

# tf.compat.v1.disable_eager_execution()
arcnlp.tf.utils.config_tf_gpu()


def build_model(model_type, data_handler, text_embedder):
    if model_type == 'bilstm':
        model = arcnlp.tf.models.BiLstmMatching(data_handler.features,
                                                data_handler.targets,
                                                text_embedder)
    elif model_type == 'esim':
        model = arcnlp.tf.models.ESIM(data_handler.features,
                                      data_handler.targets,
                                      text_embedder)
    elif model_type == 'dssm':
        model = arcnlp.tf.models.DSSM(data_handler.features,
                                      data_handler.targets,
                                      text_embedder)
    elif model_type == 'cdssm':
        model = arcnlp.tf.models.CDSSM(data_handler.features,
                                       data_handler.targets,
                                       text_embedder)
    elif model_type == 'arci':
        model = arcnlp.tf.models.ArcI(data_handler.features,
                                      data_handler.targets,
                                      text_embedder)
    elif model_type == 'arcii':
        model = arcnlp.tf.models.ArcII(data_handler.features,
                                       data_handler.targets,
                                       text_embedder)
    elif model_type == 'match_pyramid':
        model = arcnlp.tf.models.MatchPyramid(data_handler.features,
                                              data_handler.targets,
                                              text_embedder)
    elif model_type == 'knrm':
        model = arcnlp.tf.models.KNRM(data_handler.features,
                                      data_handler.targets,
                                      text_embedder)
    elif model_type == 'mvlstm':
        model = arcnlp.tf.models.MVLSTM(data_handler.features,
                                        data_handler.targets,
                                        text_embedder)
    elif model_type == 'bimpm':
        model = arcnlp.tf.models.BiMPM(data_handler.features,
                                       data_handler.targets,
                                       text_embedder)
    return model


def tokenizer(text):
    import jieba
    return jieba.lcut(text)


def run_train(args):
    dataset_builder = arcnlp.tf.data.TextMatchingData(
        arcnlp.tf.data.TextFeature(tokenizer, max_len=args.max_len),
        arcnlp.tf.data.Label())

    train_path = os.path.expanduser(args.train_path)
    val_path = os.path.expanduser(args.val_path)

    train_examples = dataset_builder.read_examples(train_path)
    dataset_builder.build_vocab(train_examples)

    train_dataset = dataset_builder.build_dataset(train_path)
    val_dataset = dataset_builder.build_dataset(val_path)

    text_embedder = tf.keras.layers.Embedding(
        len(dataset_builder.text_feature.vocab), 200, mask_zero=True)

    model = build_model(args.model_type, dataset_builder, text_embedder)

    trainer = arcnlp.tf.training.Trainer(model, dataset_builder)
    trainer.train(train_dataset=train_dataset,
                  val_dataset=val_dataset,
                  optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc'],
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
    parser.add_argument("--val_path", required=True)
    parser.add_argument("--max_len", type=int)
    parser.add_argument("--model_dir")
    parser.add_argument("--model_type", required=True,
                        choices=['bilstm', 'esim', 'dssm', 'cdssm',
                                 'arci', 'arcii', 'match_pyramid', 'knrm',
                                 'mvlstm', 'bimpm'])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    logger.info('args: %s' % args)

    run_train(args)
