import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import torch
import numpy as np
from time import time
from dgl import DGLGraph
from pandas import DataFrame
from KGEmbedUtils.kgdataUtils import dataloader

from KGEmbedUtils.ioutils import load_json_as_data_frame

# pos_emb = PositionEmbedding(emb_dim=10)
#
# idx = torch.randint(0,8, size=(1,5))
# print(idx)
#
# print(pos_emb(idx))
# # print()
# print(pos_emb(idx).shape)
# #
# rel_emd = RelationEmbedding(vocab_size=14, embed_dim=10)
# print(rel_emd(idx).shape)
# print(rel_emd)
#
# sum_emb = rel_emd(idx) + pos_emb(idx)
#
# print(sum_emb)
#
# bert_emb = BERTEmbedding(vocab_size=14, embed_dim=10)
#
# print(bert_emb(idx, idx).shape)
#
# print(bert_emb(idx, idx))

# entity_save_path = '../SeqData/Entity2PathWN18RR/'
entity_save_path = '../SeqData/Entity2PathFB_15K_237/'


def triple_for_kg_train(data: DataFrame, hop_num: int):
    def relation_path_to_numpy(row):
        in_rel_path, out_rel_path = row['i_rel_p'], row['o_rel_p']
        def path_to_numpy(path_list):
            if path_list is None:
                return (np.empty(shape=(0, hop_num), dtype=np.int32), np.empty(shape=(0,), dtype=np.int32), np.empty(shape=(0,), dtype=np.int32), 0)
            else:
                path_array = np.array(path_list, dtype=np.int32)
                path_num = len(path_list)
                paths = path_array[:,:-2]
                path_len = path_array[:,-2]
                path_freq = path_array[:,-1]
                return (paths, path_len, path_freq, path_num)
        in_paths, in_path_len, in_path_freq, in_path_num = path_to_numpy(in_rel_path)
        out_paths, out_path_len, out_path_freq, out_path_num = path_to_numpy(out_rel_path)
        center = row['i_center']
        return center, in_path_num, in_paths, in_path_len, in_path_freq, out_path_num, out_paths, out_path_len, out_path_freq

    data[['center', 'i_num', 'ir_path','ir_len', 'ir_cnt', 'o_num', 'or_path', 'or_len', 'or_cnt']] = data.apply(relation_path_to_numpy, axis=1, result_type='expand')
    rel_data = data[['center', 'i_num', 'ir_path','ir_len', 'ir_cnt', 'o_num', 'or_path', 'or_len', 'or_cnt']]
    rel_data = rel_data.astype({'center': int, 'i_num': int, 'o_num': int})
    return rel_data

# from KGEmbedUtils.ioutils import load_json_as_data_frame
#
# data: DataFrame = load_json_as_data_frame(entity_save_path + 'entity_to_tree_path_hop_3.json')
#
# rel_data = triple_for_kg_train(data, hop_num=3)
# save_to_json(rel_data, entity_save_path + 'kg_entity_to_tree_path_hop_3.json')
# #
# for idx, row in data.iterrows():
#     print(idx, row['center'])

# for col in data.columns:
#     print(col)

# def batch_to_paths(batch, data: DataFrame):
#     def gen_idexes(head_data, tail_data):
#         pos_head_in, pos_head_out = head_data['i_num'].to_numpy(), head_data['o_num'].to_numpy()
#         pos_tail_in, pos_tail_out = tail_data['i_num'].to_numpy(), tail_data['o_num'].to_numpy()
#         left_len = (pos_head_in * pos_tail_out).sum()
#         right_len = (pos_head_out * pos_tail_in).sum()
#         idx_len = left_len + right_len
#         return idx_len
#
#     pos, neg, weight, mode = batch
#     batch_size, negative_size = neg.shape[0], neg.shape[1]
#     neg_batch = neg.view(batch_size*negative_size, -1)
#
#     pos_head_data = data.iloc[pos[:,0], :]
#     pos_tail_data = data.iloc[pos[:,2], :]
#     pos_len = gen_idexes(pos_head_data, pos_tail_data)
#     print(pos_len)
#     # ##############################
#     neg_head_data = data.iloc[neg_batch[:,0], :]
#     neg_tail_data = data.iloc[neg_batch[:,2], :]
#     neg_len = gen_idexes(neg_head_data, neg_tail_data)
#     print(neg_len)
#     # neg_data = data.iloc[neg_batch, :]
#     # if mode == 'head_batch':
#     #     neg_len = gen_neg_idexes(neg_data, pos_tail_data, mode)
#     # elif mode == 'tail_batch':
#     #     neg_len = gen_neg_idexes(pos_head_data, neg_data, mode)
#     # else:
#     #     raise ValueError('Training batch mode %s not supported' % mode)
#
#
#     # print(head_data)
#     # print(tail_data)
#     # print(pos_len)
#     # if mode == 'head_batch':
#     #
#     # elif mode == 'tail_batch':
#     #
#     # else:
#     #     raise ValueError('Training batch mode %s not supported' % mode)
#     # pos_data = data.iloc[]
#

def build_graph_from_triples_directed(num_nodes: int, num_relations: int, triples, multi_graph=False) -> DGLGraph:
    """
    :param num_nodes:
    :param num_relations:
    :param triples: 3 x number of edges
    :return:
    """
    start = time()
    g = DGLGraph(multigraph=multi_graph)
    g.add_nodes(num_nodes)
    src, rel, dst = triples
    g.add_edges(src, dst)

    # ===================================================================
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    rel = torch.from_numpy(rel)
    g.ndata.update({'n_id': node_id})
    g.edata['e_label'] = rel
    # ===================================================================
    g.apply_edges(lambda edges: {'node_id_pair': torch.cat((edges.src['n_id'], edges.dst['n_id']), dim=-1)})
    # ===================================================================
    print('Constructing graph takes {:.2f} seconds'.format(time() - start))

    return g, rel
if __name__ == '__main__':
    entity_save_path = '../SeqData/Entity2PathWN18RR/'
    # entity_save_path = '../SeqData/Entity2PathFB_15K_237/'
    # data = dataloader('FB15k-237')
    # data = dataloader('WN18RR')
    data = dataloader('WN18')
    # data = dataloader('FB15K')
    num_nodes = data.num_entities
    num_relations = data.num_relation
    train_data = data.train
    relations = data.rel_dict_rev

    # print(relations)
    print("Number of entities = {}\nNumber of Edges = {}\nNumber of relations = {}.".format(num_nodes,
                                                                                            train_data.shape[0], num_relations))

    # graph, rel = build_graph_from_triples_directed(num_nodes=num_nodes,
    #                                        num_relations=num_relations,
    #                                        triples=train_data.transpose(), multi_graph=True)
    #
    # nw_graph = graph.to_networkx(node_attrs=['n_id'], edge_attrs=['e_label'])
    #
    # nx.write_gpickle(nw_graph, 'WN18.gpickle')

    # nw_graph = nx.read_gpickle('WN18.gpickle')
    # node_ids = nx.get_node_attributes(nw_graph, 'n_id')
    # edge_labels = nx.get_edge_attributes(nw_graph, 'e_label')
    # print(len(node_ids))
    # print(len(edge_labels))




#     valid_data = data.valid
#     test_data = data.test
#     print('Train: {}, valid: {}, test: {}'.format(train_data.shape[0], valid_data.shape[0], test_data.shape[0]))
#     all_data = np.concatenate([train_data, valid_data, test_data], axis=0)
#     all_true_triples = [tuple(x) for x in all_data.tolist()]
#     test_triples = [tuple(x) for x in test_data.tolist()]
#
#     # start = time()
#     # test_datat_list = construct_test_iterator(test_triples=test_triples, all_true_triples=all_true_triples, num_entity=num_nodes, num_relations=num_relations)
#     # for idx, test in enumerate(test_datat_list):
#     #     for positive_sample, negative_sample, filter_bias, mode in test:
#     #         print(idx,filter_bias[0].sum())
#     # print('Constructing time = {}'.format(time() - start))
#     train_triples = [tuple(x) for x in train_data.tolist()]
#     train_iterator = construct_train_iterator(train_triples=train_triples, num_entity=num_nodes, num_relations=num_relations, negative_sample_size=20, batch_size=4)
#     #=================================================================
    data = load_json_as_data_frame(entity_save_path + 'kg_entity_to_tree_path_hop_3.json')
#
#     # print(data[''])
#     # #=================================================================
#     for i in range(200):
#         start = time()
#         batch_i = next(train_iterator)
#         # print(batch_i[0])
#         batch_to_paths(batch_i, data)
#         # print(pos.shape, neg.shape)
#         print(time() - start)








