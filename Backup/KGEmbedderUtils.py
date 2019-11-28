import torch.nn as nn
from torch import Tensor
import math

class MLPModule(nn.Module):
    """
    Multi-layer fully-connected feed-forward neural networks
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, drop_out=0.1, num_layers=2, bias=True):
        super(MLPModule, self).__init__()
        self.relu = nn.ReLU()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim, bias=bias))
        for i in range(1, num_layers - 1):
            self.layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=bias))
        self.layers.append(nn.Linear(in_features=hidden_dim, out_features=out_dim, bias=bias))
        self.dropout = nn.Dropout(drop_out)

    def weight_init(self, m):
        # On weight initialization in deep neural networks, they state that Xavier
        # initialization is a poor choice with RELU activations
        if type(m) in [nn.Linear]:
            nn.init.kaiming_normal_(m.weight.data, a=math.sqrt(2.0))

    def forward(self, inputs: Tensor):
        h = inputs
        for i in range(0, self.num_layers - 1):
            h = self.dropout(self.relu(self.layers[i](h)))
        h = self.layers[-1](h)
        return h

class EmeddingLayer(nn.Module):
    """
    Lookup table based word embedding
    """
    def __init__(self, num_entities: int, h_dim: int, pre_trained_matrix: Tensor = None):
        super(EmeddingLayer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_entities, embedding_dim=h_dim)
        if pre_trained_matrix is not None:
            self.embedding.weight.data.copy_(pre_trained_matrix)
    def forward(self, ent_id):
        return self.embedding(ent_id)