import torch
import torch.nn as nn
from torch.autograd import Variable, Function, gradcheck
import math, numpy as np

class CosineMarginFunction(Function):
  def __init__(self, cos_k):
    self._cos_k = cos_k
    self._M = cos_k.size(0)-1

  def _find_k(self, cosine_logits):
    cos_highers = self._cos_k.data[0:self._M]
    cos_lowers = self._cos_k.data[1:]

    lower_ge = cosine_logits.unsqueeze(1).ge(cos_lowers.unsqueeze(0))
    higher_lt = cosine_logits.unsqueeze(1).lt(cos_highers.unsqueeze(0))
    k_mat = lower_ge & higher_lt
    _, k_vals = torch.max(k_mat, dim=1)
    return k_vals

  def forward(self, cosine_logits, labels):
    self._cosine_logits_size = cosine_logits.size()
    self._constant = labels

    batch_size, num_classes = cosine_logits.size()
    if cosine_logits.is_cuda:
      torch_module = torch.cuda
    else:
      torch_module = torch
    labels_onehot = torch_module.ByteTensor(batch_size, num_classes).fill_(0)
    labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
    masked_cosine_logits = torch.masked_select(cosine_logits, labels_onehot)
    masked_cosine_logits = masked_cosine_logits.clone().double()

    k_vals = self._find_k(masked_cosine_logits)
    sign_vals = torch.remainder(k_vals, 2)
    sign_vals = -1*sign_vals + 1*(1-sign_vals)
    sign_vals = sign_vals.double()
    k_vals = k_vals.double()

    float_labels_onehot = torch_module.FloatTensor(batch_size, num_classes).fill_(0)
    float_labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
    self._intermediate = masked_cosine_logits, float_labels_onehot, sign_vals

    if self._M == 1:
      amplified_cosine_logits = masked_cosine_logits
    elif self._M == 2:
      amplified_cosine_logits = (2.*masked_cosine_logits*masked_cosine_logits) - 1.
    elif self._M == 3:
      amplified_cosine_logits = masked_cosine_logits*(4.*masked_cosine_logits*masked_cosine_logits - 3.)
    elif self._M == 4:
      amplified_cosine_logits = (2.*masked_cosine_logits*masked_cosine_logits) - 1.
      amplified_cosine_logits = (2.*amplified_cosine_logits*amplified_cosine_logits) - 1.
    amplified_cosine_logits = amplified_cosine_logits*sign_vals
    amplified_cosine_logits = amplified_cosine_logits - 2*k_vals
    amplified_cosine_logits = amplified_cosine_logits.float()

    margin_cosine_logits = torch_module.FloatTensor(batch_size, num_classes)
    margin_cosine_logits = cosine_logits*(1.-float_labels_onehot)
    margin_cosine_logits.scatter_(1, labels.unsqueeze(1), amplified_cosine_logits.unsqueeze(1))
    return margin_cosine_logits

  def backward(self, grad_margin_cosine_logits):
    labels = self._constant

    batch_size, num_classes = self._cosine_logits_size
    masked_cosine_logits, float_labels_onehot, sign_vals = self._intermediate

    if grad_margin_cosine_logits.is_cuda:
      torch_module = torch.cuda
    else:
      torch_module = torch

    if self._M == 1:
      grad_amplified_cosine_logits = 1.
    elif self._M == 2:
      grad_amplified_cosine_logits = 4.*masked_cosine_logits
    elif self._M == 3:
      grad_amplified_cosine_logits = 12.*masked_cosine_logits*masked_cosine_logits - 3.
    elif self._M == 4:
      grad_amplified_cosine_logits = 16.*((2*masked_cosine_logits*masked_cosine_logits) - 1)*masked_cosine_logits
    grad_amplified_cosine_logits = grad_amplified_cosine_logits*sign_vals
    grad_amplified_cosine_logits = grad_amplified_cosine_logits.float()

    grad_cosine_logits = torch_module.FloatTensor(batch_size, num_classes)
    grad_cosine_logits = (1.-float_labels_onehot)
    grad_cosine_logits.scatter_(1, labels.unsqueeze(1), grad_amplified_cosine_logits.unsqueeze(1))
    grad_cosine_logits = grad_cosine_logits*grad_margin_cosine_logits
    return grad_cosine_logits, None, None

class CosineMargin(nn.Module):
    def __init__(self, M):
        super(CosineMargin, self).__init__()
        assert type(M)==int
        assert M in [1, 2, 3, 4]
        points = np.arange(M + 1, dtype=np.float64)
        angles = (math.pi*points)/M
        cos_values = np.cos(angles)
        self._cos_k = nn.Parameter(torch.DoubleTensor(cos_values), requires_grad=False)
        self._loss_fn = CosineMarginFunction(self._cos_k)

    def forward(self, cosine_logits, labels):
        return self._loss_fn(cosine_logits, labels)

if __name__ == "__main__":
    M = 4
    batch_size = 100
    num_classes = 20
    torch.manual_seed(99)
    logits = Variable(2*torch.rand(batch_size, num_classes).float()-1., requires_grad=True)
    labels = Variable(torch.LongTensor(batch_size).random_(num_classes))
    margin = CosineMargin(M)
    #margin.cuda()
    input_tuple = (logits, labels)
    #margin_logits = margin(*input_tuple)
    #loss_val = nn.CrossEntropyLoss()(margin_logits, labels)
    #loss_val.backward()
    test = gradcheck(margin, input_tuple, eps=1e-3, atol=1e-4)
    print test
