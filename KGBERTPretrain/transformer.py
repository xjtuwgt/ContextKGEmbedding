import torch
import torch.nn as nn
import copy
from torch import Tensor
import math
import torch.nn.functional as F

def clones(module, N):
    """Preoduce N identical layers"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    """Construct a layernorm module"""
    def __init__(self, num_features: int, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(num_features))
        self.b_2 = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x: Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.a_2 * (x - mean) /(std + self.eps) + self.b_2
        # return x

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Nodte for code similicity, the norm is first as opposed to last
    """
    def __init__(self, num_features:int, dropout: float):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(num_features=num_features)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def attention(query: Tensor, key: Tensor, value: Tensor, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.shape[-1]
    scores = torch.matmul(query, key.transpose(dim0=-2, dim1=-1))/math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    """
    Multi-head attention with 'Scaled Dot Product Attention', concat multi-head for output/residual connection
    """
    def __init__(self, h_num: int, model_dim: int, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert model_dim % h_num == 0
        self.head_dim = model_dim // h_num
        self.head_num = h_num
        self.linears = clones(nn.Linear(model_dim, model_dim), 4)
        self.dropout = nn.Dropout(p=dropout)

        for linear_i in self.linears:
            nn.init.xavier_normal_(linear_i.weight, gain=1.414)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor=None):
        if mask is not None:
            mask = mask.unsqueeze(dim=1)
        num_batches = query.shape[0]

        query, key, value = [l(x).view(num_batches, -1, self.head_num, self.head_dim)
                                 .transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        x, self.atten = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1,2).contiguous().view(num_batches, -1, self.head_num * self.head_dim)
        return self.linears[-1](x)
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, model_dim, d_hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(model_dim, d_hidden)
        self.w_2 = nn.Linear(d_hidden, model_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.w_1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.w_2.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class TransformerLayer(nn.Module):
    "TransformerLayer is made up of self-attn and feed forward (defined below)"

    def __init__(self, hidden: int, attn_heads: int, feed_forward_hidden: int, dropout: float):
        """
        :param hidden: hidden dim of transformer
        :param attn_heads: number of head
        :param feed_forward_hidden:
        :param dropout:
        """
        super(TransformerLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(h_num=attn_heads, model_dim=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(model_dim=hidden, d_hidden=feed_forward_hidden, dropout=dropout)
        self.sublayer = clones(SublayerConnection(num_features=hidden, dropout=dropout), 2)
        self.drop_out = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, self.feed_forward)
        return self.drop_out(x)