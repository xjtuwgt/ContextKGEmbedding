# import networkx as nx
from dgl import DGLGraph
import numba
from numba import jit
import numpy as np
import os
import pandas as pd
from pandas import DataFrame
from scipy import sparse
import time
import warnings
from torch import Tensor
import torch

def node_pair_with_multi_edges(self, walks: DataFrame, multi_edge_nodes: list):
    """
    There exists pair of entities between which there are multi-edge.
    :param walks:
    :param multi_edge_nodes:
    :return:
    """
    pair_in_names = []
    pair_names = []
    start = time.time()
    for idx in range(self.walklen - 1):
        walks['pair_' + str(idx)] = list(
            zip(walks['n_' + str(idx)].values.tolist(), walks['n_' + str(idx + 1)].values.tolist()))
        walks['pair_in_' + str(idx)] = walks['pair_' + str(idx)].apply(lambda x: x in multi_edge_nodes)
        pair_in_names.append('pair_in_' + str(idx))
        pair_names.append('pair_' + str(idx))
    walks['multi_edge_flag'] = walks[pair_in_names].to_numpy().sum(axis=1) > 0
    walks = walks.drop(columns=pair_in_names + pair_names)
    print('Multi-edge flag is computed in time {}'.format(time.time() - start))
    return walks


def node_walk_to_relation_walk(self, walks: DataFrame, graph: DGLGraph):
    multi_edge_nodes, _, _ = multi_edge_node_pairs(graph)
    for idx in range(self.walklen - 1):
        walks['r_' + str(idx)] = -1
    if len(multi_edge_nodes) > 0:
        start = time.time()
        walks = self.node_pair_with_multi_edges(walks, multi_edge_nodes)
        walks_groups = walks.groupby('multi_edge_flag')
        row_walks = walks_groups.get_group(True)
        row_walks = self.node_walk_to_relation_walk_by_row(row_walks, graph)
        print('Row time {}'.format(time.time() - start))
        col_walks = walks_groups.get_group(False)
        col_walks = self.node_walk_to_relation_walk_by_col(col_walks, graph)
        print('Col time {}'.format(time.time() - start))
        walks = pd.concat([row_walks, col_walks])
        walks = walks.drop(columns=['multi_edge_flag'])
        print('Total time {}'.format(time.time() - start))
    else:
        walks = self.node_walk_to_relation_walk_by_col(walks, graph)
    return walks


def node_walk_to_relation_walk_by_row(self, walks: DataFrame, graph: DGLGraph):
    from torch import Tensor
    import random
    for idx in walks.index:
        true_walk_len = walks.at[idx, 'walk_len']
        for i in range(true_walk_len):
            from_node, to_node = walks.at[idx, 'n_' + str(i)], walks.at[idx, 'n_' + str(i + 1)]
            e_id = graph.edge_id(from_node, to_node)
            if isinstance(e_id, Tensor) and e_id.shape[0] > 1:
                e_id = e_id[random.randint(0, e_id.shape[0] - 1)]
            walks.at[idx, 'r_' + str(i)] = graph.edata['e_label'][e_id].long().item()
    return walks


def node_walk_to_relation_walk_by_col(self, walks: DataFrame, graph: DGLGraph):
    groups = []
    for walk_len in range(1, self.walklen):
        mask = walks['walk_len'] == walk_len
        if mask.sum() > 0:
            group = walks.loc[mask, :].copy()
            for i in range(walk_len):
                from_idx, to_idx = group.loc[mask, 'n_' + str(i)].tolist(), group.loc[mask, 'n_' + str(i + 1)].tolist()
                edge_ids = graph.edge_ids(from_idx, to_idx)
                edge_labels = graph.edata['e_label'][edge_ids]
                group['r_' + str(i)] = edge_labels.long().tolist()
            groups.append(group)
    walks = pd.concat(groups)
    return walks


    def node_walk_to_relation_walk_fast(self, walks: DataFrame, graph: DGLGraph):
        def random_idx_reduce(edge_multi_ids: Tensor):
            """
            Mask out the n-1 redundant edges randomly.
            :param edge_multi_ids:
            :return:
            """
            mask = torch.zeros(edge_multi_ids.shape[0]).type(torch.ByteTensor)
            zero_idxes = torch.where(edge_multi_ids[:,0] == 0)[0]
            mask[zero_idxes] = 1
            print(mask.shape, mask.sum())
            dist_value, _ = np.unique(edge_multi_ids[:,0].numpy(), return_inverse=True)
            dist_value = dist_value.tolist()
            print(len(dist_value))
            for id in range(1, len(dist_value)):
                id_idxes = torch.where(edge_multi_ids[:,0]==dist_value[id])[0]
                if id_idxes.shape[0] > 1:
                    multi_edge_num = edge_multi_ids[id_idxes[0], 1]
                    node_pair_num = int(id_idxes.shape[0]/multi_edge_num)
                    id_idx_matrix = id_idxes.numpy().reshape(node_pair_num, multi_edge_num).transpose()
                    np.random.shuffle(id_idx_matrix)
                    mask[id_idx_matrix[0]] = 1
            return mask
        for idx in range(self.walklen - 1):
            walks['r_' + str(idx)] = -1
        groups = []
        for walk_len in range(1, self.walklen):
            mask = walks['walk_len'] == walk_len
            if mask.sum() > 0:
                group = walks.loc[mask, :].copy()
                for i in range(walk_len):
                    from_idx, to_idx = group.loc[mask, 'n_' + str(i)].tolist(), group.loc[mask, 'n_' + str(i + 1)].tolist()
                    edge_ids = graph.edge_ids(from_idx, to_idx)
                    if edge_ids.shape[0] == len(from_idx):
                        edge_labels = graph.edata['e_label'][edge_ids]
                        group['r_' + str(i)] = edge_labels.long().tolist()
                    else:
                        edge_labels = graph.edata['e_label'][edge_ids]
                        edge_multi_ids = graph.edata['m_edge_id'][edge_ids]
                        label_mask = random_idx_reduce(edge_multi_ids=edge_multi_ids)
                        print('mask here')
                        red_edge_labels = edge_labels[label_mask == 1]
                        group['r_' + str(i)] = red_edge_labels.long().tolist()
                groups.append(group)
        walks = pd.concat(groups)
        return walks

    relation_names = ['r_' + str(idx) for idx in range(self.walklen - 1)]
    groups = walks.groupby(relation_names)
    count = 0
    g_count = 0
    for gid, group in groups:
        # print(gid, group.shape)
        g_count = g_count + 1
        if group['rw_count'].sum() > 1:
            count = count + 1
    print(count, g_count)


    def equivalent_relation_pair_all(self, walks: DataFrame):
        walk_groups = walks.groupby('walk_pair')
        relation_names = ['r_' + str(idx) for idx in range(self.walklen - 1)]
        relation_names.append('walk_len')
        def equivalent_relation_pair(data: DataFrame):
            from itertools import combinations
            data_np = data[relation_names].to_numpy()
            data_size = data.shape[0]
            idx_list = list(range(data_size))
            combination_idx = np.array(list(combinations(idx_list,2)))
            first_idx, second_idx = combination_idx[:,0], combination_idx[:,1]
            first_data, second_data = data_np[first_idx], data_np[second_idx]
            comb_data = np.concatenate([first_data, second_data], axis=-1)
            return comb_data, data_np
        pair_list = []
        num_pairs = 0
        num_groups = 0
        for g_id, group in walk_groups:
            num_groups = num_groups + 1
            relation_pairs, sinlge_relation = equivalent_relation_pair(group)
            num_pairs = num_pairs + relation_pairs.shape[0]
            pair_list.append((g_id, relation_pairs, sinlge_relation))
        data_frame = pd.DataFrame(pair_list, columns=['group_id', 'relation_pairs', 'relations'])
        return data_frame, num_pairs, num_groups