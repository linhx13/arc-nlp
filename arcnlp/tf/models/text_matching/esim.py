# -*- coding: utf-8 -*-

import tensorflow as tf

from arcnlp_tf.models import BaseModel
from arcnlp_tf.data import DataHandler
from arcnlp_tf.layers.text_embedders import TextEmbedder
from arcnlp_tf.layers.attention import MatrixAttention


class ESIM(BaseModel):
    """ Implementation of
    `"Enhanced LSTM for Natural Language Inference"
    <https://www.semanticscholar.org/paper/Enhanced-LSTM-for-Natural-Language-Inference-Chen-Zhu/83e7654d545fbbaaf2328df365a781fb67b841b4>`
    by Chen et al., 2017.
    """

    def __init__(self,
                 data_handler: DataHandler,
                 text_embedder: TextEmbedder,
                 seq_encoder,
                 sim_func_params: Dict[str, Any] = None):
        super(ESIM, self).__init__(data_handler)

        self.text_embedder = text_embedder
        self.seq_encoder = seq_encoder
        self.matrix_attention = MatrixAttention(sim_func_params)

    def _build_model(self):
        inputs = []
        premise_input = self._create_text_inputs('premise')
        hypothesis_input = self._create_text_inputs('hypothesis')
        inputs.extend(premise_input.values())
        inputs.extend(hypothesis_input.values())
