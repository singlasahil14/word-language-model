import torch
import torch.nn as nn
from torch.autograd import Variable
from center_loss import CenterLoss

from collections import OrderedDict

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntokens, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, ALPHA=0.5):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntokens, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self._decoder = nn.Linear(nhid, ntokens)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self._decoder.weight = self.encoder.weight

        self.init_weights()

        self._cross_entropy_fn = nn.CrossEntropyLoss()
        self._center_loss_fn = CenterLoss(ntokens, nhid, ALPHA=ALPHA)

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self._decoder.bias.data.fill_(0)
        self._decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inp, hidden):
        emb = self.drop(self.encoder(inp))
        output, hidden = self.rnn(emb, hidden)
        self._embeddings = output.view(output.size(0)*output.size(1), output.size(2))
        self._drop_embeddings = self.drop(self._embeddings)
        decoded = self._decoder(self._drop_embeddings)
        return decoded, hidden

    def calculate_loss_values(self, logits, labels):
        center_loss = self._center_loss_fn(self._embeddings, labels)
        cross_entropy = self._cross_entropy_fn(logits, labels)
        loss_values = (cross_entropy, center_loss,)
        return loss_values

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
