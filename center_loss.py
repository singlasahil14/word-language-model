import torch
import torch.nn as nn
from torch.autograd import Variable, Function, gradcheck
import math, numpy as np

class CenterLossFunction(Function):
  def __init__(self, centers):
    self._centers = centers

  def forward(self, embeddings, labels):
    batch_centers = self._centers.data.index_select(0, labels)
    self._diff =  (embeddings - batch_centers)
    return torch.nn.functional.mse_loss(embeddings, batch_centers).data

  def backward(self, grad_loss):
    grad_embeddings = 2*self._diff
    grad_embeddings = (grad_embeddings*grad_loss)/self._diff.numel()
    return grad_embeddings, None

class CenterLoss(nn.Module):
    def __init__(self, num_classes, embedding_size, ALPHA=0.5, init_range=0.1):
        super(CenterLoss, self).__init__()
        self._num_classes = num_classes
        self._embedding_size = embedding_size
        self._centers = nn.Parameter(torch.Tensor(self._num_classes, self._embedding_size), requires_grad=False)
        self._centers.data.uniform_(-init_range, init_range)
        torch.nn.functional.normalize(self._centers, p=2, dim=1)
        self._ALPHA = ALPHA
        self._loss_fn = CenterLossFunction(self._centers)

    def _update_centers(self, embeddings, labels):
        diff_centers = self._centers.data.index_select(0, labels.data) - embeddings.data
        if self._centers.is_cuda:
          torch_module = torch.cuda
        else:
          torch_module = torch
        batch_size = labels.size(0)
        labels_2d = labels.clone().unsqueeze(0)
        labels_data = labels_2d.data
        label_ones = torch_module.FloatTensor(batch_size, 1).fill_(1)
        ones_size = torch.Size([self._num_classes, 1])
        label_ones_sparse = torch_module.sparse.FloatTensor(labels_data, label_ones, ones_size)
        label_ones_sparse = label_ones_sparse.coalesce()
        unique_labels = label_ones_sparse._indices()
        nonzero_count = label_ones_sparse._values()

        centers_shape = self._centers.size()
        update_centers_sparse = torch_module.sparse.FloatTensor(labels_data, diff_centers, centers_shape)
        update_centers_sparse = update_centers_sparse.coalesce()
        update_centers_dense = update_centers_sparse._values()/nonzero_count
        self._centers.data.index_add_(0, unique_labels.squeeze(0), -self._ALPHA*update_centers_dense)

    def forward(self, embeddings, labels):
        norm_vec = torch.norm(embeddings, 2, dim=1, keepdim=True)
        norm_embeddings = embeddings/norm_vec
        center_loss = self._loss_fn(norm_embeddings, labels)
        self._update_centers(norm_embeddings, labels)
        return center_loss

def main():
    batch_size = 2
    embedding_size = 4
    num_classes = 3
    torch.manual_seed(99)
    centers = Variable(torch.randn(num_classes, embedding_size).float())
    embeddings = nn.Parameter(torch.randn(batch_size, embedding_size).float())
    labels = Variable(torch.LongTensor(batch_size).random_(num_classes))
    input_tuple = (embeddings, labels)
    center_loss_fn = CenterLossFunction(centers)

    test = gradcheck(center_loss_fn, input_tuple, eps=1e-3, atol=1e-4)
    print test

    center_loss = CenterLoss(num_classes, embedding_size)
    loss_val = center_loss(*input_tuple)
    loss_val.backward()

if __name__ == "__main__":
    main()

