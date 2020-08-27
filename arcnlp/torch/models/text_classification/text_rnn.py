from torch import nn


class TextRNN(nn.Module):

    def __init__(self, num_classes, token_embedder, rnn_type="lstm",
                 hidden_size=128, num_layers=2, dropout=0.1):
        super(TextRNN, self).__init__()
        self.token_embedder = token_embedder
        if rnn_type == 'lstm':
            rnn_cls = nn.LSTM
        elif rnn_type == 'gru':
            rnn_cls = nn.GRU
        else:
            raise ValueError("rnn_type %s is invalid" % rnn_type)
        self.encoder = rnn_cls(token_embedder.embedding_dim,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=True,
                               bidirectional=True,
                               dropout=dropout)
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, tokens):
        # TODO: pack_padded_sequence
        embedded = self.token_embedder(tokens)
        encoded, _ = self.encoder(embedded)
        encoded = encoded[:, -1, :]
        if self.dropout:
            encoded = self.dropout(encoded)
        return self.fc(encoded)
