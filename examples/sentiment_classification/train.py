# -*- coding: utf-8 -*-

import logging

import tensorflow as tf
import arcnlp.tf

logger = logging.getLogger(__name__)


def build_model(model_type, data_handler, text_embedder):
    if model_type == 'bow':
        model = arcnlp.tf.models.BOWClassifier(data_handler.features,
                                               data_handler.targets,
                                               text_embedder)
    elif model_type == 'text_cnn':
        model = arcnlp.tf.models.TextCNNClassifier(data_handler.features,
                                                   data_handler.targets,
                                                   text_embedder)
    elif model_type == 'bilstm':
        model = arcnlp.tf.models.BiLSTMClassifier(data_handler.features,
                                                  data_handler.targets,
                                                  text_embedder)
    elif model_type == 'rcnn':
        model = arcnlp.tf.models.RCNNClassifier(data_handler.features,
                                                data_handler.targets,
                                                text_embedder)
    return model


def run_train(args):
    token_fields = {
        'word': arcnlp.tf.data.Field()
    }
    data_handler = arcnlp.tf.data.FasttextDataHandler(token_fields)
    train_dataset, test_dataset = arcnlp.tf.utils.create_train_test_datasets(
        data_handler, args.train_path, args.test_path, args.test_size)
    data_handler.build_vocab(train_dataset, test_dataset)
    text_embedder = arcnlp.tf.layers.text_embedders.BasicTextEmbedder({
        'word': tf.keras.layers.Embedding(len(token_fields['word'].vocab), 200)
    })

    model = build_model(args.model_type, data_handler, text_embedder)

    trainer = arcnlp.tf.training.Trainer(model,
                                         data_handler,
                                         optimizer='adam',
                                         loss='categorical_crossentropy',
                                         metrics=['acc'])
    trainer.train(train_dataset=train_dataset,
                  validation_dataset=test_dataset,
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
    parser.add_argument("--model_dir")
    parser.add_argument("--model_type", required=True,
                        choices=['bow', 'text_cnn', 'bilstm', 'rcnn'])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    logger.info('args: %s' % args)

    run_train(args)
