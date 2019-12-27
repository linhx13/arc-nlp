# -*- coding: utf-8 -*-

import os
import pickle
import logging
import json
from shutil import copyfile
from typing import Tuple

import tensorflow as tf
import numpy as np

from ..data import DataHandler, Dataset, DataGenerator
# from ..models import BaseModel, model_classes
from .. import utils

logger = logging.getLogger(__name__)

TASK_CONFIG_FILE = "task_config.json"
DATA_HANDLER_FILE = "data_handler.pkl"
MODEL_CONFIG_FILE = "model_config.json"
MODEL_FILE = "model.h5"
CHECKPOINT_MODEL_FILE = "model.epoch_{epoch:d}.h5"
EPOCH_MODLE_FILE = 'model.epoch_%d.h5'


class Trainer(object):

    def __init__(self,
                 model: tf.keras.models.Model,
                 data_handler: DataHandler,
                 optimizer='adam',
                 loss="categorical_crossentropy",
                 metrics=["acc"]):
        self.model = model
        self.data_handler = data_handler
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    @classmethod
    def from_path(cls, path: str, epoch: int = None):
        model, data_handler = load_model_data(path, epoch)
        return cls(model, data_handler)

    def train(self,
              train_dataset: Dataset,
              validation_dataset: Dataset = None,
              validation_split: float = 0.1,
              validation_metric="val_acc",
              batch_size: int = 32,
              epochs: int = 20,
              patience: int = 3,
              data_gen_type: str = 'sequence',
              model_dir: str = None,
              **fit_kwargs):
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        self.model.summary()

        if model_dir:
            utils.mkdir_p(model_dir)

        if model_dir:
            task_config = {}
            task_config['model'] = {}
            task_config['model']['class_name'] = self.model.__class__.__name__
            with open(os.path.join(model_dir, TASK_CONFIG_FILE), 'w') as fp:
                json.dump(task_config, fp)
            with open(os.path.join(model_dir, DATA_HANDLER_FILE), 'wb') as fp:
                pickle.dump(self.data_handler, fp)
            with open(os.path.join(model_dir, MODEL_CONFIG_FILE), 'w') as fp:
                fp.write(self.model.to_json())

        training_data = self._create_data(train_dataset, batch_size,
                                          data_gen_type)
        if validation_dataset:
            validation_data = self._create_data(
                validation_dataset, batch_size, data_gen_type, train=False)
        else:
            validation_data = None

        callbacks = self._get_callbacks(validation_metric, patience,
                                        model_dir)
        fit_kwargs.update({
            "epochs": epochs,
            "batch_size": batch_size,
            "callbacks": callbacks
        })
        if validation_dataset is not None:
            fit_kwargs['validation_data'] = validation_data
        elif validation_split > 0.0 and data_gen_type == 'arrays':
            fit_kwargs['validation_split'] = validation_split

        if data_gen_type == 'arrays':
            history = self.model.fit(training_data[0], training_data[1],
                                     **fit_kwargs)
        elif data_gen_type == 'sequence':
            fit_kwargs.pop("batch_size")
            # fit_kwargs['use_multiprocessing'] = True
            history = self.model.fit(training_data, **fit_kwargs)

        if model_dir:
            best_epoch = int(np.argmax(history.history[validation_metric]))
            self._save_best_model(best_epoch, model_dir)

        return history

    def evaluate(self,
                 dataset: Dataset,
                 batch_size: int = 32,
                 data_gen_type: str = 'sequence'):
        data = self._create_data(dataset, batch_size, data_gen_type,
                                 train=False)
        if data_gen_type == 'arrays':
            score = self.model.evaluate(data[0], data[1],
                                        batch_size=batch_size)
        elif data_gen_type == 'sequence':
            score = self.model.evaluate_generator(data)
        return dict(zip(self.model.metrics_names, score))

    def _get_callbacks(self, validation_metric, patience, model_dir):
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor=validation_metric, patience=patience)
        # TODO: add model_callbacks
        callbacks = [early_stop]
        if model_dir:
            checkpoint_path = os.path.join(
                model_dir, CHECKPOINT_MODEL_FILE)
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                save_best_only=True,
                save_weights_only=False,
                monitor=validation_metric)
            callbacks.append(model_checkpoint)
        return callbacks

    def _save_best_model(self, best_epoch, model_dir):
        ckpt_path = os.path.join(
            model_dir, EPOCH_MODLE_FILE % (best_epoch + 1))
        model_path = os.path.join(model_dir, MODEL_FILE)
        copyfile(ckpt_path, model_path)
        logger.info("Saved the best model to %s" % model_path)

    def _create_data(self, dataset: Dataset, batch_size: int,
                     data_gen_type: str, train: bool = True):
        data_generator = DataGenerator(self.data_handler)
        if data_gen_type == 'arrays':
            return data_generator.create_data_arrays(dataset, batch_size,
                                                     train)
        elif data_gen_type == 'sequence':
            return data_generator.create_data_sequence(dataset, batch_size,
                                                       train)
        else:
            raise ValueError("Invalid data_gen_type: %s" % self.data_gen_type)

    def _get_task_config(self):
        config = {}
        config['model'] = {}
        config['model']['class_name'] = self.model.__class__.__name__


def load_model_data(model_dir, epoch: int = None) \
        -> Tuple[tf.keras.models.Model, DataHandler]:
    logger.info("Loading data handler ...")
    with open(os.path.join(model_dir, DATA_HANDLER_FILE), 'rb') as fp:
        data_handler = pickle.load(fp)
    logger.info("Loading model ...")
    with open(os.path.join(model_dir, TASK_CONFIG_FILE)) as fp:
        task_config = json.load(fp)
    custom_objects = utils.get_custom_objects()
    if epoch is not None:
        model_file = EPOCH_MODLE_FILE % epoch
    else:
        model_file = MODEL_FILE
    model_file = os.path.join(model_dir, model_file)
    model = tf.keras.models.load_model(model_file, custom_objects)
    logger.info("Loading model from %s done" % model_file)
    return model, data_handler
