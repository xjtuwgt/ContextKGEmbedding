import torch.nn as nn

class RelationEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_dim=128):
        super().__init__(num_embeddings=vocab_size + 3, embedding_dim=embed_dim)