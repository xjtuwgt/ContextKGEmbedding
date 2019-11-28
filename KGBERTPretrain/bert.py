import torch.nn as nn
from Embedding.bert_emb import BERTEmbedding
from .transformer import TransformerLayer

class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size: int, max_rel_len: int, hidden, n_layers, head_num, drop_out=0.1):
        """
        :param vocab_size:
        :param hidden:
        :param n_layers:
        :param attn_heads:
        :param drop_out:
        """
        super(BERT, self).__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.head_num = head_num
        self.feed_forward_hidden = hidden * 4

        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_dim=hidden, max_len=max_rel_len)
        self.transformer_bolcks = nn.ModuleList(
            [TransformerLayer(hidden=hidden, attn_heads=head_num,
                              feed_forward_hidden=self.feed_forward_hidden, dropout=drop_out) for _ in range(n_layers)])

    def forward(self, rel_seq, pos_seq, seq_mask):
        x = self.embedding(rel_seq, pos_seq)
        mask = seq_mask.unsqueeze(1).repeat(1, seq_mask.size(1), 1)
        for transformer in self.transformer_bolcks:
            x = transformer.forward(x, mask)
        return x