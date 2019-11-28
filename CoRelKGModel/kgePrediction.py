import torch
import torch.nn as nn
import logging
from torch import Tensor
from KGBERTPretrain.bert import BERT
import math
import numpy as np

class FeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, model_dim, d_hidden, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(model_dim, d_hidden)
        self.w_2 = nn.Linear(d_hidden, 1)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)

    def forward(self, x):
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))

class ContextKGEModel(nn.Module):
    def __init__(self, num_entity, num_relation, hidden_dim, gamma, bert_path_encoder:BERT, pool_type='attn', drop_out=0.1):
        """
        :param num_entity:
        :param num_relation:
        :param path_len:
        :param hidden_dim:
        :param gamma: Margin based loss
        :param bert_path_encoder:
        :param pool_type:
        """
        super(ContextKGEModel, self).__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.bert_path_encoder = bert_path_encoder
        for param in self.bert_path_encoder.parameters(recurse=True):
            param.requires_grad = False
        self.score_function = FeedForward(model_dim=hidden_dim, d_hidden=4*hidden_dim)
        # self.score_function = nn.Linear(in_features=hidden_dim, out_features=1, bias=True)
        self.sfa_matrix=nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.sfa_matrix.weight)
        self.attn_pool_matrix = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=False)
        self.attn_score_function = nn.Linear(in_features=hidden_dim, out_features=1, bias=False)
        self.pool_type = pool_type
        self.drop_out = nn.Dropout(p=drop_out)
        self.sigmoid = torch.sigmoid

    def forward(self, sample):
        trip_paths, path_pos, path_mask, tri2path_size = sample
        print('Paths {}'.format(trip_paths[0]))
        print('Position {}'.format(path_pos[0]))
        print('Mask {}'.format(path_mask[0]))
        path_emb = self.path_embedding(trip_paths=trip_paths, path_pos=path_pos, path_mask=path_mask)
        # path_emb = torch.rand(path_emb.shape).to(path_emb.device)
        path_emb_pool, batch_size, neg_samp_size = self.path_set_pool(triple_emb=path_emb[:,0], tri2path_size=tri2path_size)
        scores = self.score(trip_path_emb=path_emb_pool)
        scores = self.sigmoid(scores.view(batch_size, (neg_samp_size+1)))
        scores = scores.view(batch_size, (neg_samp_size + 1))
        p_scores, n_scores = scores[:,0], scores[:,1:]
        loss = torch.relu((n_scores + self.gamma).sub(p_scores[:, None]))
        loss = loss.sum()
        return loss

    def path_embedding(self, trip_paths, path_pos, path_mask):
        path_embeddings = self.bert_path_encoder(trip_paths, path_pos, path_mask)
        return path_embeddings

    def score(self, trip_path_emb):
        scores = self.score_function(trip_path_emb)
        return scores

    def model_test(self, sample):
        trip_paths, path_pos, path_mask, tri2path_size = sample
        path_emb = self.path_embedding(trip_paths=trip_paths, path_pos=path_pos, path_mask=path_mask)
        path_emb_pool, batch_size, neg_samp_size = self.path_set_pool(triple_emb=path_emb[:, 0],
                                                                      tri2path_size=tri2path_size)
        scores = self.score(trip_path_emb=path_emb_pool)
        scores = scores.view(batch_size, (neg_samp_size + 1))
        p_scores, n_scores = scores[:, 0], scores[:, 1:]
        return p_scores, n_scores

    def path_set_pool(self, triple_emb, tri2path_size):
        path_emb_set_list, batch_size, neg_sample_size = self.data_split(data=triple_emb, path_tri_nums=tri2path_size)
        if self.pool_type == 'mean':
            path_emb_pool = torch.stack([self.avergage_pool(x) for x in path_emb_set_list], dim=0)
        elif self.pool_type == 'min':
            path_emb_pool = torch.stack([self.min_pool(x) for x in path_emb_set_list], dim=0)
        elif self.pool_type == 'max':
            path_emb_pool = torch.stack([self.max_pool(x) for x in path_emb_set_list], dim=0)
        elif self.pool_type == 'attn':
            path_emb_pool = torch.stack([self.self_attentive_pooling(x) for x in path_emb_set_list], dim=0)
            # path_emb_pool = torch.stack([self.attentive_pooling(x) for x in path_emb_set_list], dim=0)
        else:
            raise ValueError('Pooling mode %s not supported' % self.pool_type)
        return path_emb_pool, batch_size, neg_sample_size

    @staticmethod
    def avergage_pool(data: Tensor):
        return torch.mean(data, dim=0)

    @staticmethod
    def max_pool(data: Tensor):
        return torch.max(data, dim=0)[0]

    @staticmethod
    def min_pool(data: Tensor):
        return torch.min(data, dim=0)[0]

    def self_attentive_pooling(self, data: Tensor):
        data = self.drop_out(data)
        ap_data = self.sfa_matrix(data)
        G = torch.tanh(torch.mm(data, ap_data.t()))
        max_g = torch.max(G, dim=0)[0]
        att_weight = torch.softmax(max_g,dim=0)
        att_weight = self.drop_out(att_weight)
        res = torch.mm(att_weight.unsqueeze(0), data)
        return res

    def attentive_pooling(self, data: Tensor):
        ap_data = self.attn_pool_matrix(data)
        att_weight = self.attn_score_function(ap_data)
        att_weight = torch.softmax(att_weight,dim=0)
        res = torch.mm(att_weight.view(1,-1), data)
        return res

    @staticmethod
    def data_split(data: Tensor, path_tri_nums: Tensor):
        batch_size, neg_samp_size = path_tri_nums.shape[0], path_tri_nums.shape[1] - 1
        trip_split_sizes = tuple(path_tri_nums.view(batch_size * (neg_samp_size + 1)).tolist())
        res = torch.split(data, split_size_or_sections=trip_split_sizes, dim=0)
        return res, batch_size, neg_samp_size

    @staticmethod
    def train_step(model, optimizer, optim_scheduler, train_iterator, device, regularization: float=0.0):
        model.train()
        optimizer.zero_grad()
        batch = {key: value.to(device) for key, value in next(train_iterator).items()}
        paths, path_mask, path_position, path_batch_num, path_total_num, weight = batch['path'], batch['path_mask'], \
                                                                                        batch['path_position'], batch['path_bias_idx'], \
                                                                                        batch['path_num'], batch['batch_weight'] #, batch['mode']

        batch_i = (paths,  path_position, path_mask, path_batch_num)
        loss = model(batch_i)
        # if regularization !=0.0:
        #     regularization_term = regularization * model.score_function.weight.norm(p=3) ** 3
        #     loss = loss + regularization_term
        optim_scheduler.zero_grad()
        loss.backward()
        optim_scheduler.step_and_update_lr()
        # optimizer.step()
        torch.cuda.empty_cache()
        return loss



