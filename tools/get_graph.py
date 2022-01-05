import torch
import torch.nn as nn
from torch_geometric.nn import knn_graph


class RandomDrop(nn.Module):
    """
    Find dilated neighbor from neighbor list
    edge_index: (2, batch_size, num_points, k)
    """
    def __init__(self, k=20, random_rate=1.0, stochastic=True, isTrain=True):
        super(RandomDrop, self).__init__()
        self.random_rate = random_rate
        self.stochastic = stochastic
        self.k = k
        self.isTrain = isTrain

    def forward(self, edge_index):
        if self.stochastic:
            num = int(self.k * self.random_rate)
            if self.isTrain:
                randnum = torch.randperm(self.k)[:num]
                edge_index = edge_index[:, :, :, randnum]
            else:
                edge_index = edge_index[:, :, :, :num]
        return edge_index


class GetKnnGraph(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """
    def __init__(self, k=20, stochastic=True, random_rate=1.0, isTrain=True):
        super(GetKnnGraph, self).__init__()
        self.stochastic = stochastic
        self.k = k
        self._random = RandomDrop(k, random_rate, stochastic, isTrain)
        self.knn = knn_graph
        self.random_rate = random_rate

    def forward(self, x):
        x = x.squeeze(-1)
        B, C, N = x.shape
        edge_index = []
        for i in range(B):
            edgeindex = self.knn(x[i].contiguous().transpose(1, 0).contiguous(), self.k)
            if edgeindex.size(1) != N * self.k:
                edgeindex = edgeindex[:, :N * self.k]
            edgeindex = edgeindex.view(2, N, self.k)
            edge_index.append(edgeindex)
        edge_index = torch.stack(edge_index, dim=1)
        return self._random(edge_index)
