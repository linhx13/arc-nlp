# -*- coding: utf-8 -*-

import logging

import arcnlp.tf

logger = logging.getLogger(__name__)


def run_evaluate(model_dir, data_path):
    arcnlp.tf.utils.config_tf_gpu()
    trainer = arcnlp.tf.training.Trainer.from_path(model_dir)
    dataset = trainer.data_handler.build_dataset_from_path(data_path)
    eval_res = trainer.evaluate(dataset)
    print(eval_res)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s %(asctime)s %(filename)s:%(lineno)d %(message)s')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--data_path", required=True)
    args = parser.parse_args()
    logger.info('args: %s' % args)

    run_evaluate(args.model_dir, args.data_path)
