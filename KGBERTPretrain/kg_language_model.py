import torch.nn as nn
from .bert import BERT

class BERTLM(nn.Module):
    """
    Relation path pair prediction + Masked Language SavedModel
    """

    def __init__(self, bert: BERT, rel_vocab_size: int, num_class=2):
        """
        :param bert:
        :param rel_vocab_size:
        """
        super(BERTLM, self).__init__()
        self.bert = bert
        self.relation_pair = RelationPairPrediction(hidden=self.bert.hidden, num_class=num_class)
        self.mask_lm = MaskedLanguageModel(hidden=self.bert.hidden, vocab_size=rel_vocab_size)

    def forward(self, rel_seq, pos_seq, seq_mask):
        x = self.bert(rel_seq, pos_seq, seq_mask)
        return self.relation_pair(x), self.mask_lm(x)

class RelationPairPrediction(nn.Module):
    """
    4-class classification/2-class classification
    """

    def __init__(self, hidden, num_class):
        """
        :param hidden:
        """
        super(RelationPairPrediction, self).__init__()
        self.linear = nn.Linear(hidden, num_class)
        nn.init.xavier_normal_(self.linear.weight, gain=1.414)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))


class MaskedLanguageModel(nn.Module):
    """
    predicting origin relation from masked input sequence
    n-class classification problem, n-class = rel_vocab_size
    """
    def __init__(self, hidden, vocab_size):
        super(MaskedLanguageModel, self).__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        nn.init.xavier_normal_(self.linear.weight, gain=1.414)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))