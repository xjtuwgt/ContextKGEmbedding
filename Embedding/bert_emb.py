import torch.nn as nn
from .relation import RelationEmbedding
from .position import PositionEmbedding, PositionEmbeddingTrainable
class BERTEmbedding(nn.Module):
    """
        BERT Embedding which is consisted of
        1. TokenEmbedding : embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos

        sum of all these features to form BERT embedding
    """
    def __init__(self, vocab_size, embed_dim, max_len, drop_out=0.1):
        super(BERTEmbedding, self).__init__()
        self.rel_emb = RelationEmbedding(vocab_size=vocab_size, embed_dim=embed_dim)
        # self.pos_emb = PositionEmbedding(emb_dim=embed_dim, max_len=max_len)
        self.pos_emb = PositionEmbeddingTrainable(embed_dim=embed_dim, max_len=max_len)
        self.drop_out = nn.Dropout(p=drop_out)
        self.embed_dim = embed_dim

    def forward(self, rel_seq, pos_seq):
        rel_seq_emb = self.rel_emb(rel_seq)
        pos_seq_emb = self.pos_emb(pos_seq)
        seq_emb = rel_seq_emb + pos_seq_emb
        return self.drop_out(seq_emb)
