# -*- coding: utf-8 -*-

from typing import Dict, List

from tensorflow.keras.models import Model
import numpy as np

from ..data import DataHandler, Example
from ..utils import config_tf_gpu
from ..training import load_model_data


class Predictor(object):

    def __init__(self, model: Model, data_handler: DataHandler) -> None:
        self.model = model
        self.data_handler = data_handler

    @classmethod
    def from_path(cls, model_dir: str, **gpu_config_kwargs):
        config_tf_gpu(**gpu_config_kwargs)
        model, data_handler = load_model_data(model_dir)
        return cls(model=model, data_handler=data_handler)

    def predict_json(self, data: Dict, **kwargs) -> Dict:
        return self.predict_batch_json([data], **kwargs)[0]

    def predict_batch_json(self, inputs: List[Dict], **kwargs) -> List[Dict]:
        examples = [self._json_to_example(data) for data in inputs]
        features = {n: f.process([getattr(ex, n) for ex in examples])
                    for n, f in self.data_handler.features.items()}
        preds = self.model.tf_model.predict(features, **kwargs)
        outputs = [{} for _ in inputs]
        for name, batch_decoded in self..decode(preds).items():
            for output, decoded in zip(outputs, batch_decoded):
                output[name] = decoded
        return outputs

    def _json_to_example(self, json_dict) -> Example:
        raise NotImplementedError

    def decode(self, preds: np.ndarray) -> Dict[str, np.ndarry]:
        return {'preds': preds}
