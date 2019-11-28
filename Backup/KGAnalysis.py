import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Backup.KGUtils import build_graph_from_triples_directed
from dgl import DGLGraph
from KGEmbedUtils.kgdataUtils import dataloader
from KGEmbedUtils.ioutils import *


def multi_tail_head(g: DGLGraph):
    """
    :param g:
    :return: 1-to-N, N-to-1 and N-to-N
    """
    def send_edge_message(edges):
        return {'m_type': edges.data['e_label']} # get the edge labels

    def type_reduce_func(nodes):
        types = nodes.mailbox['m_type']
        batch_size, neighbor_num = types.shape[0], types.shape[1]
        multi_types = torch.zeros(batch_size)
        for i in range(0, batch_size):
            unique_i = types[i,:].unique()
            unique_num = unique_i.shape.numel()
            if unique_num != neighbor_num:
                multi_types[i] = 1
        return {'multi_flag': multi_types}

    def edge_func(edge):
        src_id, dst_id = edge.src['n_id'], edge.dst['n_id']
        edge_num_flag = torch.zeros(src_id.shape[0])
        for i in range(0, src_id.shape[0]):
            if g.edge_ids(src_id[i], dst_id[i]).shape.numel() > 1:
                edge_num_flag[i] = 1
        return {'edge_flag': edge_num_flag}

    g.apply_edges(edge_func)
    edge_flag_sum = g.edata.pop('edge_flag').sum()
    # node_pairs = g.edata['node_id_pair']
    # node_pairs = node_pairs[edge_flag_sum == 1, :]
    print('N-to-N: Multi-relations number = {}/{} with ratio = {:.4f}'
          .format(edge_flag_sum, g.number_of_edges(), edge_flag_sum/g.number_of_edges()))


    g_r = g.reverse(share_edata=True)
    g.register_reduce_func(type_reduce_func)
    g.register_message_func(send_edge_message)
    g.update_all(message_func=send_edge_message, reduce_func=type_reduce_func)

    g_r.register_reduce_func(type_reduce_func)
    g_r.register_message_func(send_edge_message)
    g_r.update_all(message_func=send_edge_message, reduce_func=type_reduce_func)

    g_flag, g_r_flag = g.ndata.pop('multi_flag'), g_r.ndata.pop('multi_flag')
    print('N-to-1: {}\n1-to-N: {}\nNode number: {}'.format(g_flag.sum(), g_r_flag.sum(), g.number_of_nodes()))
    print('Ratio N-to-1 {:.4f}\nRatio 1-to-N {:.4f}'.format(g_flag.sum()/g.number_of_nodes(),
                                                                     g_r_flag.sum()/g.number_of_nodes()))

def multi_tail_head_distribution(g: DGLGraph, rel_num: int, rel_dict: dict):
    def send_edge_message(edges):
        return {'m_type': edges.data['e_label']} # get the edge labels

    def type_reduce_func(nodes):
        types = nodes.mailbox['m_type'] # get the edge labels
        batch_size, neighbor_num = types.shape[0], types.shape[1] # batch size (number of nodes) x neighbor numbers
        multi_types = torch.zeros(batch_size, rel_num, dtype=torch.long)
        for i in range(0, batch_size): # for each entity
            values, frequents = torch.unique(types[i,:], return_counts=True) #value = relation, frequent = number of relations
            multi_types[i, values] = frequents # set the frequency of each appeared relation
        return {'type_dist': multi_types}


    g.register_reduce_func(type_reduce_func)
    g.register_message_func(send_edge_message)
    g.update_all(message_func=send_edge_message, reduce_func=type_reduce_func)
    freq_dist = g.ndata.pop('type_dist')
    multi_heads = torch.sum(freq_dist > 1)
    relations = torch.sum(freq_dist > 0)

    max_rel_idx = torch.argmax(freq_dist)
    row = max_rel_idx//g.number_of_nodes() -1
    col = max_rel_idx % rel_num
    print('Relation with max N-to-1 {} = {}'.format(torch.max(freq_dist), rel_dict[freq_dist[row, col].data.item()]))
    freq_dist_array = freq_dist[freq_dist > 0]
    sum_freq = torch.sum(freq_dist_array)
    print('Sum = {} # (relation, tail) = {} and average = {}'.format(sum_freq, relations,
                                                                     sum_freq.float() / relations))
    freq_uniques = torch.unique(freq_dist_array, sorted=True, return_counts=True)
    print('Unique relation number = {}'.format(freq_uniques[0].shape))
    for i in range(freq_uniques[0].shape[0]):
        print('index = {}, value = {}, count = {}, ratio = {:.2f}'.format(i + 1, freq_uniques[0][i],
                                                                          freq_uniques[1][i],
                                                                          freq_uniques[1][i]*100/sum_freq.float()))
    print('Sorted = {}'.format(freq_dist_array.sort(descending=True)[0]))
    print('N-to-1 relations: {}, total = {}'.format(multi_heads, relations))

    #============================================================================
    g_r = g.reverse(share_edata=True)
    g_r.register_reduce_func(type_reduce_func)
    g_r.register_message_func(send_edge_message)
    g_r.update_all(message_func=send_edge_message, reduce_func=type_reduce_func)

    freq_dist = g_r.ndata.pop('type_dist')
    multi_heads = torch.sum(freq_dist > 1)
    relations = torch.sum(freq_dist > 0)
    max_rel_idx = torch.argmax(freq_dist)
    row = max_rel_idx//g.number_of_nodes() - 1
    col = max_rel_idx % rel_num
    print('Relation with max N-to-1 {} = {}'.format(torch.max(freq_dist), rel_dict[freq_dist[row, col].data.item()]))

    freq_dist_array = freq_dist[freq_dist > 0]
    sum_freq = torch.sum(freq_dist_array)
    print('Sum = {} # (relation, tail) = {} and average = {}'.format(sum_freq, relations,
                                                                     sum_freq.float() / relations))
    freq_uniques = torch.unique(freq_dist_array, sorted=True, return_counts=True)
    print('Distinct relation number = {}'.format(freq_uniques[0].shape))
    for i in range(freq_uniques[0].shape[0]):
        print('index = {}, value = {}, count = {}, ratio = {:.2f}'.format(i + 1, freq_uniques[0][i],
                                                                          freq_uniques[1][i],
                                                                          freq_uniques[1][i] * 100 / sum_freq.float()))
    print('Sorted = {}'.format(freq_dist_array.sort(descending=True)[0]))
    print('1-to-N relations: {}, total = {}'.format(multi_heads, relations))

def relation_distribution(graph: DGLGraph):
    """
    Relation distribution on the graph
    :param graph:
    :return:
    """
    relations = graph.edata['e_label']
    r_unique = relations.unique(sorted=True)
    r_unique_count = torch.stack([(relations == r_u).sum() for r_u in r_unique])

    r_uni_count = torch.cat([r_unique.unsqueeze(dim=-1),
                             r_unique_count.unsqueeze(dim=-1)], dim=-1)
    print(r_uni_count)

def triangles(graph: DGLGraph):
    """
    Compute the number of triangles in the graph
    :param graph:
    :return:
    """
    import networkx as nx
    start = time()
    network_graph = graph.to_networkx(node_attrs=['n_id'], edge_attrs=['e_label'])
    print('Converting to network graph takes {} seconds'.format(time() - start))
    start = time()
    network_graph = network_graph.to_undirected()
    cliq_list = list(nx.clique.enumerate_all_cliques(network_graph))
    print('Extracting all triangles takes {} seconds'.format(time() - start))
    triangle_num = 0
    triangles = []
    for cliq in cliq_list:
        if len(cliq) == 3:
            triangle_num = triangle_num + 1
            triangles.append(cliq)
    print(triangle_num)
    return triangles


# def WNEntityEmbeddingBall(embeding_file_name, radius=2):
#     data = WN18RR()
#     num_nodes = data.num_nodes
#     num_relations = data.num_rels
#     train_triples, val_triples, test_triples = data.train.transpose(), data.valid.transpose(), data.test.transpose()
#     graph, rel, in_degree, out_degree = build_graph_from_triples_directed(num_nodes=num_nodes,
#                                            num_relations=num_relations,
#                                            triples=train_triples)
#
#     # ==============================================================================
#     rand_seed = 1234
#     torch.manual_seed(rand_seed)
#     np.random.seed(rand_seed)
#     # ==============================================================================
#     entityEmbeder = EntityEmbedding(graph, radius=radius)
#
#     ##==================================
#     emb_parameters = entityEmbeder.embedding.weight.numpy()
#     np.savez(embeding_file_name, embeding=emb_parameters)
#     ##==================================
#
#
# def FB237EntityEmbeddingBall(embeding_file_name, radius=2):
#     data = FB15k237()
#     num_nodes = data.num_nodes
#     num_relations = data.num_rels
#     train_triples, val_triples, test_triples = data.train.transpose(), data.valid.transpose(), data.test.transpose()
#     graph, rel, in_degree, out_degree = build_graph_from_triples_directed(num_nodes=num_nodes,
#                                            num_relations=num_relations,
#                                            triples=train_triples)
#     rand_seed = 1234
#     torch.manual_seed(rand_seed)
#     np.random.seed(rand_seed)
#     # ==============================================================================
#     entityEmbeder = EntityEmbedding(graph, radius=radius)
#
#     ##==================================
#     emb_parameters = entityEmbeder.embedding.weight.numpy()
#     np.savez(embeding_file_name, embeding=emb_parameters)
#     ##==================================

# def unique_embedding(embedding_file_name):
#     ent_embedding = np.load(embedding_file_name)
#     data = ent_embedding['embeding']
#     arr_dict = dict()
#     idx = 0
#     for i in range(data.shape[0]):
#         arr_i = np.array2string(data[i])
#         if arr_i not in arr_dict:
#             arr_dict[arr_i] = idx
#             idx = idx + 1
#         else:
#             print(i)
#     print((ent_embedding['embeding']).shape, len(arr_dict), len(arr_dict) * 1.0/data.shape[0])
#

if __name__ == '__main__':
    path = '../Data/EntityEmbeder/'
    # data = dataloader('FB15k-237')
    data = dataloader('WN18RR')
    # data = dataloader('FB15K')
    num_nodes = data.num_entities
    num_relations = data.num_relation
    train_data = data.train
    relations = data.rel_dict_rev
    # print(relations)
    print("Number of entities = {}\nNumber of Edges = {}\nNumber of relations = {}.".format(num_nodes, train_data.shape[0], num_relations))
    graph, rel, in_degree, out_degree, multi_edge_nodes = build_graph_from_triples_directed(num_nodes=num_nodes,
                                           num_relations=num_relations,
                                           triples=train_data.transpose())

    from_ids = [28313, 23262, 14959]
    to_ids = [1378, 28313, 14960]
    e_ids = graph.edge_ids(from_ids, to_ids)
    print(graph.edata['e_label'][e_ids])

    # # print(np.unique(rel.numpy(), return_counts=True))
    #
    # # print((graph.edata['e_label'].shape))
    # # print(len(multi_edge_nodes[0]), len(multi_edge_nodes[1]), len(multi_edge_nodes[2]))
    #
    # # print(out_degree[out_degree > 0].shape)
    # # import networkx as nx
    # # #++++++++++++++++++++++++++++++++++++
    # from RandomWalk.RandWalkOnGraph import KG2RelPattern
    # seed = 2019
    # set_seeds(seed)
    # epochs = 4000
    # walk_len = 8
    # seq_data_path = '../SeqData/RandWalk_WN18RR/'
    # g2v = KG2RelPattern(epochs=epochs, walklen=walk_len, walklen_var_flag=True)
    # walks, relation_walks = g2v.extractor(graph)
    # print(walks.shape, relation_walks.shape)
    # # # ++++++++++++++++++++++++++++++++++++
    # walk_data_file_name = 'Walks_Epoch_'+str(epochs) + "_walklen_"+str(walk_len) + '_seed_' + str(seed)
    # rel_pair_data_file_name = 'Relation_Walk_Epoch_' + str(epochs) + "_walklen_" + str(walk_len) + '_seed_' + str(seed)
    # save_to_HDF(walks, seq_data_path + walk_data_file_name + '.h5')
    # save_to_json(walks, seq_data_path + walk_data_file_name + '.json')
    # save_to_json(relation_walks, seq_data_path + rel_pair_data_file_name + '.json')
    # # df = load_json_as_data_frame(seq_data_path + walk_data_file_name + ".json")
    # # df = load_HDF_as_data_frame(seq_data_path + walk_data_file_name + '.h5')
    # # df = load_json_as_data_frame(seq_data_path + rel_pair_data_file_name + ".json")
    # # print(df.shape)