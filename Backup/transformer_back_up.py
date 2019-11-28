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

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, num_features, self_attn: MultiHeadedAttention,
                 feed_forward: PositionwiseFeedForward, dropout: float):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(num_features=num_features, dropout=dropout), 2)
        self.num_features = num_features

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, encoder_layer: EncoderLayer, num_layer: int):
        super(Encoder, self).__init__()
        self.layers = clones(encoder_layer, num_layer)
        self.norm = LayerNorm(encoder_layer.num_features)

    def encode(self, input_ids, input_mask):
        return self.encoder(self.embedder(input_ids), input_mask)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Embeddings(nn.Module):
    def __init__(self, vocab_size: int, num_features: int):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(num_embeddings=vocab_size, embedding_dim=num_features)
        self.num_features = num_features

    def forward(self, input_idx):
        return self.lut(input_idx) * math.sqrt(self.num_features)

##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class EncoderModel(nn.Module):
    def __init__(self, encoder: Encoder, embedder: Embeddings):
        super(EncoderModel, self).__init__()
        self.encoder = encoder
        self.embedder = embedder

    def forward(self, input, input_mask):
        return self.encoder(self.embedder(input), input_mask)

def make_model(vocab, num_layers=6, num_features=512, hidden_dim=2048, num_head=8, dropout=0.1):

    attn = MultiHeadedAttention(h_num=num_head, model_dim=num_features, dropout=dropout)
    feed_forward = PositionwiseFeedForward(model_dim=num_features, d_hidden=hidden_dim, dropout=dropout)
    deep_copy = copy.deepcopy
    embedder = Embeddings(vocab_size=vocab, num_features=num_features)
    encoder = Encoder(EncoderLayer(num_features=num_features,
                                 self_attn=deep_copy(attn),
                                 feed_forward=deep_copy(feed_forward),
                                 dropout=dropout), num_layer=num_layers)
    model = EncoderModel(encoder=encoder, embedder=embedder)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

if __name__ == '__main__':
    # torch.manual_seed(2019)
#     # x = torch.rand(2,3,4)
#     # print(x)
#     # norm_layer = LayerNorm(4)
#     # print(norm_layer(x))
    encoder = make_model(100)