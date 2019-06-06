# -*- coding: utf-8 -*-

import logging

import numpy as np
from keras.callbacks import EarlyStopping, LambdaCallback, ModelCheckpoint
from keras import backend as K

from .. import utils

logger = logging.getLogger(__name__)


MODEL_CONFIG_FILE = "model_config.json"
MODEL_WEIGHTS_FILE = "model_weights.h5"
CHECKPOINT_WEIGHTS_FILE = "model_weights.epoch_{epoch:d}.h5"


class Trainer(object):

    def __init__(self, model,
                 train_dataset, validation_dataset=None,
                 validation_split=0.1, uses_data_generator=True,
                 optimizer="adam", loss="categorical_crossentropy",
                 metrics=["acc"], validation_metric="val_acc",
                 batch_size=32, num_epoches=20, patience=3,
                 serialization_dir=None, **fit_kwargs):
        self.model = model

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.validation_split = validation_split
        self.uses_data_generator = uses_data_generator

        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.validation_metric = validation_metric
        self.batch_size = batch_size
        self.num_epoches = num_epoches
        self.patience = patience

        self.serialization_dir = serialization_dir
        self.fit_kwargs = fit_kwargs

        self.best_epoch = -1

    def train(self):
        logger.info("Start training ...")
        if self.serialization_dir:
            utils.mkdir_p(self.serialization_dir)

        logger.info("Compiling the model ...")
        self.model.summary()
        compile_kwargs = self._compile_kwargs()
        self.model.compile(**compile_kwargs)
        if self.serialization_dir:
            model_json_file = os.path.join(self.serialization_dir,
                                           MODEL_CONFIG_FILE)
            with open(model_json_file, 'w') as fout:
                fout.write(self.model.to_json())
            logger.info("Saved model config json to %s" % model_json_file)

        logger.info("Building model fit params ...")
        callbacks = self._get_callbacks()
        fit_kwargs = {"epochs": self.num_epoches, "batch_size": self.batch_size,
                      "callbacks": callbacks}

        if self.validation_dataset is not None:
            fit_kwargs['validation_data'], validation_iter = \
                self._create_tensors(self.validation_dataset)
            if self.uses_data_generator:
                fit_kwargs['validation_steps'] = len(validation_iter)
        elif self.validation_split > 0.0 and not self.uses_data_generator:
            fit_kwargs['validation_split'] = self.validation_split
        fit_kwargs.update(self.fit_kwargs)

        train_arrays, train_iter = fuck()

        logger.info("Fitting the model ...")
        if not self.uses_data_generator:
            history = self.model.fit(train_arrays[0], train_arrays[1],
                                     **fit_kwargs)
        else:
            fit_kwargs.pop('batch_size')
            fit_kwargs['steps_per_epoch'] = len(train_iter)
            history = self.model.fit_generator(train_arrays, **fit_kwargs)

        logger.info("Saving best model ...")
        self.best_epoch = \
            int(np.argmax(history.history[self.validation_metric]))
         if self.serialization_dir:
            self._save_best_model()

    def _compile_kwargs(self):
        return {
            "optimizer": self.optimizer,
            "loss": self.loss,
            "metrics": self.metrics
        }

    def _get_callbacks(self):
        early_stop = EarlyStopping(monitor=self.validation_metric)
