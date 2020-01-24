# -*- coding: utf-8 -*-

import os
import pickle
import logging
from shutil import copyfile
from typing import Tuple

import tensorflow as tf
import numpy as np

from ..data import DataHandler, Dataset
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
              validation_metric="val_acc",
              batch_size: int = 32,
              epochs: int = 20,
              patience: int = 3,
              model_dir: str = None,
              **fit_kwargs):
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        self.model.summary()

        if model_dir:
            utils.mkdir_p(model_dir)

        if model_dir:
            with open(os.path.join(model_dir, DATA_HANDLER_FILE), 'wb') as fp:
                pickle.dump(self.data_handler, fp)
            with open(os.path.join(model_dir, MODEL_CONFIG_FILE), 'w') as fp:
                fp.write(self.model.to_json())

        training_data = self.data_handler.get_data_sequence(
            train_dataset, batch_size, train=True)
        if validation_dataset:
            validation_data = self.data_handler.get_data_sequence(
                validation_dataset, batch_size, train=False)
        else:
            validation_data = None

        callbacks = self._get_callbacks(validation_metric, patience,
                                        model_dir)
        fit_kwargs.update({
            "epochs": epochs,
            "callbacks": callbacks
        })
        if validation_dataset is not None:
            fit_kwargs['validation_data'] = validation_data
        # fit_kwargs['use_multiprocessing'] = True
        history = self.model.fit(training_data, **fit_kwargs)

        if model_dir:
            best_epoch = int(np.argmax(history.history[validation_metric]))
            self._save_best_model(best_epoch, model_dir)

        return history

    def evaluate(self,
                 dataset: Dataset,
                 batch_size: int = 32):
        data = self.data_handler.get_data_sequence(dataset, batch_size, False)
        score = self.model.evaluate(data)
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


def load_model_data(model_dir, epoch: int = None) \
        -> Tuple[tf.keras.models.Model, DataHandler]:
    logger.info("Loading data handler ...")
    with open(os.path.join(model_dir, DATA_HANDLER_FILE), 'rb') as fp:
        data_handler = pickle.load(fp)
    logger.info("Loading model ...")
    custom_objects = utils.get_custom_objects()
    if epoch is not None:
        model_file = EPOCH_MODLE_FILE % epoch
    else:
        model_file = MODEL_FILE
    model_file = os.path.join(model_dir, model_file)
    model = tf.keras.models.load_model(model_file, custom_objects)
    logger.info("Loading model from %s done" % model_file)
    return model, data_handler
