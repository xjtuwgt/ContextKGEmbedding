import numpy as np
from random import random as rand

##        for pos in cand_pos[:num_pred]:
            # masked_rels.append(relation_pair[pos])
            # masked_pos.append(pos)
            # if rand() < 0.8:
            #     relation_pair[pos] = MASK
            # elif rand() < 0.5:
            #     relation_pair[pos] = get_random_relation(pure_relation_vocab, relation_pair[pos])

#
# def path_position_mask(paths, MASK, relation_vocab, num_pred=1):
#     m, n = paths.shape[0], paths.shape[1]
#     path_pad = paths.copy()
#     path_pad[path_pad < 0] = 0
#     path_mask = (paths >= 0).astype(np.int16)
#     path_pos = np.argsort(-path_mask)
#     first_idxs, path_position = np.repeat(np.array(np.arange(m)).reshape(m, 1), n, axis=1), np.repeat(
#         np.array(np.arange(n)).reshape(1, n), m, axis=0)
#     temp_idx = np.zeros(paths.shape, dtype=np.int16)
#     temp_idx[first_idxs, path_pos] = path_position
#     path_pos = path_position[first_idxs, temp_idx]
#     path_pos[paths < 0] = 0
#     #===========================================
#     def get_random_relation(relation_vocab, relation):
#         rand_rel = np.random.choice(relation_vocab)
#         while rand_rel == relation:
#             rand_rel = np.random.choice(relation_vocab)
#         return rand_rel
#     mask_rels, mask_pos, mask_mask = np.zeros(shape=(m, num_pred), dtype=np.int32), \
#                                      np.zeros(shape=(m, num_pred), dtype=np.int32), np.zeros(shape=(m, num_pred), dtype=np.int32)
#     in_path_rel_mask = ((paths >=0) & (paths <11)).astype(np.int16)
#     path_lens = in_path_rel_mask.sum(axis=-1)
#     for i in range(m):
#         if path_lens[i] > num_pred:
#             rel_idxes = np.where(in_path_rel_mask[i]==1)[0]
#             mask_pos[i] = rel_idxes[np.random.choice(len(rel_idxes), size=num_pred, replace=False)]
#             mask_rels[i] = paths[i, mask_pos[i]]
#             mask_mask[i] = 1
#             for k in range(num_pred):
#                 mask_pos_k = mask_pos[i,k]
#                 if rand() < 0.8:
#                     path_pad[i, mask_pos_k] = MASK
#                 elif rand() < 0.5:
#                     path_pad[i, mask_pos_k] = get_random_relation(relation_vocab=relation_vocab, relation=path_pad[i, mask_pos_k])
#     #===========================================
#     return path_pad, path_mask, path_pos, mask_rels, mask_pos, mask_mask
#
# paths = np.array([[12,0,-1,2,-1,11],[12,1,-1,2,4,11],[12,-1,-1,2,-1,11]])
#
# a, b, c, d, e, f = path_position_mask(paths, 0, 11)
#
# print(paths)
# print(a)
# print(b)
# print(c)
# print(d)
# print(e)
# print(f)
import torch
import torch.nn.functional as F
from torch import Tensor
import math
from KGEmbedUtils.utils import set_seeds
def attention(query: Tensor, key: Tensor, value: Tensor, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.shape[-1]
    scores = torch.matmul(query, key.transpose(dim0=-2, dim1=-1))/math.sqrt(d_k)
    print(scores)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        print(scores)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

set_seeds(100)
x = torch.rand(1,3,4)
x_mask = torch.Tensor([1,0,1]).view(1,3)
x_mask = x_mask.unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(dim=1)
print(x_mask)
y = attention(x, x, x, mask=x_mask)
print(y)

