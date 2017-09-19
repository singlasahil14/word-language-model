import torch
import torch.nn as nn
from torch.autograd import Variable
from center_loss import CenterLoss
from cosine_margin import CosineMargin

from collections import OrderedDict

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntokens, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, ALPHA=0.5, BETA=128, normalize_weights=False, zero_bias=False):
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

        self.BETA = BETA
        self._normalize_weights = normalize_weights
        self._use_bias = not(zero_bias)
        self._decoder = nn.Linear(nhid, ntokens, bias=self._use_bias)

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
        self._one_margin_fn = CosineMargin(1)
        self._two_margin_fn = CosineMargin(2)
        self._three_margin_fn = CosineMargin(3)
        self._four_margin_fn = CosineMargin(4)
        self._cosine_margin_fn_dict = OrderedDict([(1, self._one_margin_fn), 
                                                   (2, self._two_margin_fn),
                                                   (3, self._three_margin_fn), 
                                                   (4, self._four_margin_fn)])

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if self._decoder.bias is not None:
            self._decoder.bias.data.fill_(0)
        self._decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inp, hidden):
        emb = self.drop(self.encoder(inp))
        output, hidden = self.rnn(emb, hidden)
        self._embeddings = output.view(output.size(0)*output.size(1), output.size(2))
        self._drop_embeddings = self.drop(self._embeddings)
        if self._normalize_weights:
            self._decoder.weight.data = torch.nn.functional.normalize(self._decoder.weight.data)
        decoded = self._decoder(self._drop_embeddings)
        return decoded, hidden

    def calculate_loss_values(self, logits, labels):
        center_loss = self._center_loss_fn(self._embeddings, labels)

        embeddings_norm = torch.norm(self._drop_embeddings, 2, dim=1, keepdim=True)
        weight_norm = torch.norm(self._decoder.weight, 2, dim=1, keepdim=True)
        total_norm = embeddings_norm*weight_norm.t()
        if self._use_bias:
            nobias_logits = logits - self._decoder.bias
        else:
            nobias_logits = logits
        cosine_logits = nobias_logits/total_norm
        margin_cross_entropy_values = []
        cosine_margin_fn_list = self._cosine_margin_fn_dict.values()
        for margin_fn in cosine_margin_fn_list:
            margin_cosine_logits = margin_fn(cosine_logits, labels)
            margin_logits = total_norm*margin_cosine_logits
            margin_logits = margin_logits + self._decoder.bias
            ce_logits = (margin_logits + logits*self.BETA)/(1+self.BETA)
            margin_cross_entropy = self._cross_entropy_fn(ce_logits, labels)
            margin_cross_entropy_values.append(margin_cross_entropy)
        loss_values = tuple(margin_cross_entropy_values) + (center_loss,)
        return loss_values

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
