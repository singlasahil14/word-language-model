import torch
import torch.nn as nn
from torch.autograd import Variable, Function, gradcheck
import math, numpy as np

class CenterLossFunction(Function):
  def __init__(self, centers, ALPHA=0.5):
    self._num_classes, self._embedding_size = centers.size()
    self._centers = centers
    self._ALPHA = ALPHA

  def forward(self, embeddings, labels):
    batch_centers = self._centers.data.index_select(0, labels)
    self._diff =  embeddings - batch_centers
    self._labels = labels.unsqueeze(0)
    return torch.nn.functional.mse_loss(embeddings, batch_centers).data

  def backward(self, grad_loss):
    grad_embeddings = 2*self._diff
    grad_embeddings = (grad_embeddings*grad_loss)/self._diff.numel()
    if grad_loss.is_cuda:
      torch_module = torch.cuda
    else:
      torch_module = torch
    label_ones = torch_module.FloatTensor(self._labels.t().size()).fill_(1)
    ones_size = torch.Size([self._num_classes, 1])
    label_ones_sparse = torch_module.sparse.FloatTensor(self._labels, label_ones, ones_size)
    label_ones_sparse = label_ones_sparse.coalesce()
    nonzero_count = label_ones_sparse._values()

    centers_shape = self._centers.size()
    update_centers_sparse = torch_module.sparse.FloatTensor(self._labels, self._diff, centers_shape)
    update_centers_sparse = update_centers_sparse.coalesce()
    unique_labels = update_centers_sparse._indices()
    update_centers_dense = update_centers_sparse._values()/nonzero_count
    update_centers_sparse = torch_module.sparse.FloatTensor(unique_labels, update_centers_dense, centers_shape)
    self._centers.data.add(self._ALPHA*update_centers_sparse)
    return grad_embeddings, None

class CenterLoss(nn.Module):
    def __init__(self, num_classes, embedding_size, ALPHA=0.5, init_range=0.1):
        super(CenterLoss, self).__init__()
        self._centers = nn.Parameter(torch.Tensor(num_classes, embedding_size), requires_grad=False)
        self._centers.data.uniform_(-init_range, init_range)
        self._loss_fn = CenterLossFunction(self._centers, ALPHA=ALPHA)

    def forward(self, embeddings, labels):
        return self._loss_fn(embeddings, labels)

def main():
    batch_size = 2
    embedding_size = 4
    num_classes = 3
    torch.manual_seed(99)
    embeddings = Variable(torch.randn(batch_size, embedding_size).float(), requires_grad=True)
    labels = Variable(torch.LongTensor(batch_size).random_(num_classes))
    input_tuple = (embeddings, labels)
    center_loss = CenterLoss(num_classes, embedding_size)

    test = gradcheck(center_loss, input_tuple, eps=1e-3, atol=1e-4)
    print test

if __name__ == "__main__":
    main()
