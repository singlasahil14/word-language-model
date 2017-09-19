import torch
import torch.nn as nn
from torch.autograd import Variable, Function, gradcheck
import math, numpy as np

class CenterLossFunction(Function):
  def __init__(self, centers):
    self._centers = centers

  def forward(self, embeddings, labels):
    embeddings_d = embeddings.double()
    batch_centers = self._centers.data.index_select(0, labels)
    self._diff = embeddings_d - batch_centers
    return torch.nn.functional.mse_loss(embeddings_d, batch_centers).data

  def backward(self, grad_loss):
    grad_embeddings = 2*self._diff/self._diff.numel()
    grad_embeddings = (grad_embeddings*grad_loss)
    return grad_embeddings, None

class CenterLoss(nn.Module):
    def __init__(self, num_classes, embedding_size, ALPHA=0.5, init_range=0.1):
        super(CenterLoss, self).__init__()
        self._num_classes = num_classes
        self._embedding_size = embedding_size
        self._centers = nn.Parameter(torch.rand(num_classes, embedding_size).double(), requires_grad=False)
        self._centers.data.uniform_(-init_range, init_range)
        self._centers.data = torch.nn.functional.normalize(self._centers.data)
        self._ALPHA = ALPHA

    def _update_centers(self, embeddings, batch_centers, labels):
        if self._centers.is_cuda:
          torch_module = torch.cuda
        else:
          torch_module = torch
        batch_size = labels.size(0)
        labels_2d = labels.clone().unsqueeze(0)
        labels_data = labels_2d.data
        label_ones = torch_module.DoubleTensor(batch_size, 1).fill_(1)
        ones_size = torch.Size([self._num_classes, 1])
        label_ones_sparse = torch_module.sparse.DoubleTensor(labels_data, label_ones, ones_size)
        label_ones_sparse = label_ones_sparse.coalesce()
        unique_labels = label_ones_sparse._indices().squeeze(0)
        nonzero_count = label_ones_sparse._values()

        centers_shape = self._centers.size()
        embeddings_centers_sparse = torch_module.sparse.DoubleTensor(labels_data, embeddings.data, centers_shape)
        embeddings_centers_sparse = embeddings_centers_sparse.coalesce()
        embeddings_centers_dense = embeddings_centers_sparse._values()/nonzero_count
        unique_centers = self._centers.data.index_select(0, unique_labels)
        updated_centers = unique_centers - self._ALPHA*(unique_centers - embeddings_centers_dense)
        updated_centers = torch.nn.functional.normalize(updated_centers)
        self._centers.data.index_copy_(0, unique_labels, updated_centers)

    def forward(self, embeddings, labels):
        embeddings_d = embeddings.double()
        norm_vec = torch.norm(embeddings_d, 2, dim=1, keepdim=True)
        batch_centers = torch.nn.functional.embedding(labels, self._centers)
        scaled_batch_centers = batch_centers*norm_vec
        center_loss = torch.nn.functional.mse_loss(embeddings_d, scaled_batch_centers)
        norm_embeddings = embeddings_d/norm_vec
        self._update_centers(norm_embeddings, batch_centers, labels)
        return center_loss.float()

def main():
    batch_size = 3
    embedding_size = 8
    num_classes = 4
    torch.manual_seed(99)
    centers = Variable(torch.randn(num_classes, embedding_size).double())
    embeddings = nn.Parameter(torch.randn(batch_size, embedding_size).float())
    labels = Variable(torch.LongTensor(batch_size).random_(num_classes))
    input_tuple = (embeddings, labels)
    center_loss_fn = CenterLossFunction(centers)

#    test = gradcheck(center_loss_fn, input_tuple, eps=1e-3, atol=1e-4)
#    print test

    center_loss = CenterLoss(num_classes, embedding_size)
    loss_val = center_loss(*input_tuple)
    loss_val.backward()

if __name__ == "__main__":
    main()

