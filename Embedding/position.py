import torch.nn as nn
import torch
import math

class PositionEmbedding(nn.Module):
    """
    Position embedding
    """
    def __init__(self, emb_dim, max_len=9):
        super(PositionEmbedding, self).__init__()
        pos_emb = torch.zeros(max_len, emb_dim).float()
        pos_emb.requires_grad = False #fixed the position embedding

        positions = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, emb_dim, 2).float() * -(math.log(10000.0) / emb_dim)).exp()
        pos_emb[:,0::2] = torch.sin(positions * div_term)
        pos_emb[:,1::2] = torch.cos(positions * div_term)
        self.register_buffer('pos_emb', pos_emb)

    def forward(self, p_idxs):
        return self.pos_emb[p_idxs]


class PositionEmbeddingTrainable(nn.Embedding):
    def __init__(self, embed_dim, max_len=9):
        super().__init__(num_embeddings=max_len, embedding_dim=embed_dim)