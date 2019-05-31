# -*- coding: utf-8 -*-

class Predictor(object):
    def __init__(self, model, dataset_reader):
        self.model = model
        self.dataset_reader = dataset_reader

    def predict_instance(self, instance):
        pass

    def predict_json(self, inputs):
        instance = self._json_to_instance(inputs)
        return predict_instance(instance)

    def _json_to_instance(self, inputs):
        raise NotImplementedError