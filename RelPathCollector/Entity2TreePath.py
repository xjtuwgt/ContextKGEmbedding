from time import time
from dgl.contrib.sampling import NeighborSampler
from dgl.nodeflow import NodeFlow
from pandas import DataFrame
import pandas as pd
from torch import Tensor
from Backup.KGUtils import build_graph_from_triples_directed
from dgl import DGLGraph
import numpy as np
import torch
from KGEmbedUtils.kgdataUtils import dataloader
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)


def tree_extractor(g: DGLGraph, hop_num: int, num_workers=8):
    """
    :param g: whole graph
    :param radius: radius of each ball (bi-directional)
    :return: a set of sub-graphs (each ball is extracted for each node)
    """
    g.readonly(readonly_state=True)
    def NodeFlow2Tree(nf: NodeFlow):
        center = nf.layer_parent_nid(-1)[0].item()
        nodes_list = [[] for _ in range(0, hop_num)]
        nodes_list[0] = (1, set(nf.layer_parent_nid(hop_num - 1).tolist()))
        union_set = nodes_list[0][1]
        for i in range(1, hop_num):
            node_set_i = set(nf.layer_parent_nid(hop_num - i - 1).tolist())
            node_set_i_diff = node_set_i.difference(union_set)
            nodes_list[i] = ((i + 1, node_set_i_diff))
            union_set = union_set.union(node_set_i_diff)
        return center, nodes_list
    expand_factor = g.number_of_nodes()
    out_tree_list = []
    for nf_out in NeighborSampler(g=g, expand_factor=expand_factor, batch_size=1, neighbor_type='out',
                              shuffle=False, num_hops=hop_num, num_workers=num_workers):
        center, out_nodes = NodeFlow2Tree(nf_out)
        out_tree_list.append((center, out_nodes))
        # print('out', center, out_nodes)
    out_tree = pd.DataFrame(out_tree_list, columns=['center', 'out_nodes'])
    in_tree_list = []
    for nf_in in NeighborSampler(g=g, expand_factor=expand_factor, batch_size=1, neighbor_type='in',
                                 shuffle=False, num_hops=hop_num, num_workers=num_workers):
        center, int_nodes = NodeFlow2Tree(nf_in)
        # print('in', center.item(), int_nodes)
        in_tree_list.append((center, int_nodes))
    in_tree = pd.DataFrame(in_tree_list, columns=['center', 'in_nodes'])
    tree = pd.merge(left=out_tree, right=in_tree, on='center', how='inner')
    g.readonly(readonly_state=False)
    return tree

def tree_to_shortest_path(tree_df: DataFrame, hop_num, graph: DGLGraph):
    """
    :param tree_df: center, in_nodes, out_nodes
    :return:
    """
    def path_row(row):
        center, in_nodes, out_nodes = row['center'], row['in_nodes'], row['out_nodes']
        def shortest_path(center, red_node_list, neighbor_type: str):
            if len(red_node_list[0][1]) == 0:
                return [], set(), np.full((0, hop_num + 2), fill_value=-1, dtype=np.int64)
            shortest_path_list = [[] for _ in range(len(red_node_list))]
            leaf_node_dict = dict()
            for x in red_node_list[0][1]:
                shortest_path_list[0].append((center, x))
                leaf_node_dict[x] = (0, 0)

            if len(red_node_list) > 1:
                for i in range(1, len(red_node_list)):
                    for entity in red_node_list[i][1]:
                        leaf_node_dict[entity] = (0, i)
                        for pre_path in shortest_path_list[i-1]:
                            pre_entity = pre_path[-1]
                            if neighbor_type == 'out':
                                if graph.has_edge_between(pre_entity, entity):
                                    shortest_path_list[i].append(pre_path + (entity,))
                                    leaf_node_dict[pre_entity] = (leaf_node_dict[pre_entity][0] + 1, leaf_node_dict[pre_entity][1])
                                    break
                                else:
                                    continue
                            else:
                                if graph.has_edge_between(entity, pre_entity):
                                    shortest_path_list[i].append(pre_path + (entity,))
                                    leaf_node_dict[pre_entity] = (leaf_node_dict[pre_entity][0] + 1, leaf_node_dict[pre_entity][1])
                                    break
                                else:
                                    continue
            leaf_node_set = set([(key, value[1]) for key, value in leaf_node_dict.items() if value[0] == 0])
            leaf_paths = np.full((len(leaf_node_set), hop_num + 2), fill_value=-1, dtype=np.int64)
            for idx, leaf in enumerate(leaf_node_set):
                leaf_node, leaf_layer_id = leaf[0], leaf[1]
                for path in shortest_path_list[leaf_layer_id]:
                    if path[-1] == leaf_node:
                        leaf_paths[idx,:len(path)] = path
                        leaf_paths[idx, hop_num + 1] = len(path) - 1
                        break
            return shortest_path_list, leaf_node_set, leaf_paths

        _, in_leaf_node_set, in_leaf_paths = shortest_path(center, in_nodes, neighbor_type='in')
        _, out_leaf_node_set, out_leaf_paths = shortest_path(center, out_nodes, neighbor_type='out')
        return in_leaf_paths, out_leaf_paths, in_leaf_node_set, out_leaf_node_set
    tree_df[['in_leaf_path', 'out_leaf_path', 'in_leaf', 'out_leaf']] = tree_df.parallel_apply(path_row, axis=1, result_type="expand")
    tree_df = pd.DataFrame(tree_df[['center', 'in_leaf_path', 'out_leaf_path']])
    return tree_df

def entity_path_to_relation(data: DataFrame, graph: DGLGraph, hop_num: int):
    in_tree_paths = np.concatenate(data['in_leaf_path'].tolist()).astype(np.int64)
    out_tree_paths = np.concatenate(data['out_leaf_path'].tolist()).astype(np.int64)
    in_tree_df = pd.DataFrame(in_tree_paths, columns=(['n_'+str(i) for i in range(hop_num + 1)] + ['walk_len']))
    out_tree_df = pd.DataFrame(out_tree_paths, columns=(['n_'+str(i) for i in range(hop_num + 1)] + ['walk_len']))

    def node_walk_to_relation_walk_fast(walks: DataFrame, graph: DGLGraph, hop_num: int, neighbor_type: str):
        def random_idx_reduce(edge_multi_ids: Tensor):
            """
            Mask out the n-1 redundant edges randomly.
            :param edge_multi_ids:
            :return:
            """
            mask = torch.zeros(edge_multi_ids.shape[0]).type(torch.ByteTensor)
            edge_multi_df = pd.DataFrame(edge_multi_ids.numpy(), columns=['edge_idx', 'edge_count'])
            edge_multi_df['idx'] = edge_multi_df.index
            multi_edge_groups = edge_multi_df.groupby('edge_idx')
            for g_id, group in multi_edge_groups:
                if g_id == 0:
                    mask[group['idx'].to_numpy()] = 1
                else:
                    multi_edge_num = group['edge_count'].values[0]
                    node_pair_num = int(group.shape[0] / multi_edge_num)
                    id_idx_matrix = group['idx'].to_numpy().reshape(node_pair_num, multi_edge_num).transpose()
                    np.random.shuffle(id_idx_matrix)
                    mask[id_idx_matrix[0]] = 1
            return mask

        for idx in range(hop_num):
            walks['r_' + str(idx)] = -1
        groups = []
        for walk_len in range(1, hop_num + 1):
            mask = walks['walk_len'] == walk_len
            if mask.sum() > 0:
                group = walks.loc[mask, :].copy()
                for i in range(walk_len):
                    from_idx, to_idx = group.loc[mask, 'n_' + str(i)].tolist(), group.loc[
                        mask, 'n_' + str(i + 1)].tolist()
                    if neighbor_type == 'out':
                        edge_ids = graph.edge_ids(from_idx, to_idx)
                    else:
                        edge_ids = graph.edge_ids(to_idx, from_idx)
                    if edge_ids.shape[0] == len(from_idx):
                        edge_labels = graph.edata['e_label'][edge_ids]
                        group['r_' + str(i)] = edge_labels.long().tolist()
                    else:
                        edge_labels = graph.edata['e_label'][edge_ids]
                        edge_multi_ids = graph.edata['m_edge_id'][edge_ids]
                        label_mask = random_idx_reduce(edge_multi_ids=edge_multi_ids)
                        red_edge_labels = edge_labels[label_mask == 1]
                        group['r_' + str(i)] = red_edge_labels.long().tolist()
                groups.append(group)
        walks = pd.concat(groups)
        return walks

    # # print(in_tree_df)
    # # print(out_tree_df)
    #
    # #193, 9021, tensor([41059, 41060])
    # print('hello there', graph.edge_ids(193, 9021))

    out_tree_df = node_walk_to_relation_walk_fast(out_tree_df, graph=graph, hop_num=hop_num, neighbor_type='out')
    in_tree_df = node_walk_to_relation_walk_fast(in_tree_df, graph=graph, hop_num=hop_num, neighbor_type='in')

    def group_rel_paths(tree_df: DataFrame, neighbor_type: str, hop_num: int):
        paths = [[i, None, None] for i in range(data.shape[0])]
        rel_col_name = ['r_'+str(i) for i in range(hop_num)]
        ent_col_name = ['n_'+str(i) for i in range(hop_num + 1)]
        if neighbor_type == 'in':
            ent_col_name = ent_col_name[::-1]
            rel_col_name = rel_col_name[::-1]
        rel_col_name.append('walk_len')
        ent_col_name.append('walk_len')
        groups = tree_df.groupby('n_0')
        group_number = 1
        for gid, group in groups:
            if group_number % 2000 == 0:
                print('Processing group {}'.format(group_number))
            rel_paths = group[rel_col_name]
            rel_paths_with_count = rel_paths.groupby(rel_col_name).size().reset_index(name='r_count')
            paths[gid][1] = rel_paths_with_count.to_numpy() #relation based path, path_length, count
            paths[gid][2] = group[ent_col_name].to_numpy()  #entity_based_path, path_length
            group_number = group_number + 1
        paths = [tuple(x) for x in paths]
        return paths

    in_ent_paths = group_rel_paths(in_tree_df, neighbor_type='in', hop_num=hop_num)
    in_ent_tree_data = pd.DataFrame(in_ent_paths, columns=['i_center', 'i_rel_p', 'i_ent_p'])

    out_ent_paths = group_rel_paths(out_tree_df, neighbor_type='out', hop_num=hop_num)

    out_ent_tree_data = pd.DataFrame(out_ent_paths, columns=['o_center', 'o_rel_p', 'o_ent_p'])
    tree = pd.concat([in_ent_tree_data, out_ent_tree_data], axis=1)
    return tree

def triple_for_kg_train(data: DataFrame, hop_num: int):
    def relation_path_to_numpy(row):
        in_rel_path, out_rel_path = row['i_rel_p'], row['o_rel_p']
        def path_to_numpy(path_list):
            if path_list is None:
                return (np.empty(shape=(0, hop_num), dtype=np.int32), np.empty(shape=(0,), dtype=np.int32), np.empty(shape=(0,), dtype=np.int32), 0)
            else:
                path_array = np.array(path_list, dtype=np.int32)
                path_num = len(path_list)
                if path_num != path_array.shape[0]:
                    print('here')
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

def entity_to_paths(graph: DGLGraph, hop_num: int, num_workers=8):
    start = time()
    print('Step 1: Tree extracting...')
    graph_copy_for_samp = graph_copy(graph=graph)
    tree_df = tree_extractor(g=graph_copy_for_samp, hop_num=hop_num, num_workers=num_workers)

    print('\tTree generation of {} nodes in {:.2f} seconds'.format(graph.number_of_nodes(), time() - start))
    print('Step 2: Shortest path extracting...')
    start = time()
    entity_path_df = tree_to_shortest_path(tree_df=tree_df, graph=graph, hop_num=hop_num)

    print('\tShortest generation of {} nodes in {:.2f} seconds'.format(graph.number_of_nodes(), time() - start))
    print('Step3: Entity path to relation path...')
    start = time()
    tree_to_path = entity_path_to_relation(data=entity_path_df, graph=graph, hop_num=hop_num)
    print('\tEntity path transformation in {:.2f} seconds'.format(time() - start))

    for idx, row in tree_to_path.iterrows():
        if idx <= 1:
            for col in tree_to_path.columns:
                print(row[col])

    print('Step4: Entity path to in/out relation path...')
    start = time()
    tree_to_rel_path = triple_for_kg_train(data=tree_to_path, hop_num=hop_num)
    print('\tTree to relation path set in {:.2f} seconds'.format(time() - start))
    return tree_to_path, tree_to_rel_path



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

    # from_nodes = [0, 5363]
    # to_nodes = [5363, 31860]
    # edge_ids = graph.edge_ids(from_nodes, to_nodes)
    # print(graph.edata['e_label'][edge_ids])
    # # print(graph.edata['e_label'][])


    from KGEmbedUtils.utils import graph_copy
    # print(graph.edata)
    # print(graph.edges())
    # g = graph_copy(graph=graph)
    # # print(in_degree[0], out_degree[0])
    entity_save_path = '../SeqData/Entity2PathWN18RR/'
    # entity_save_path = '../SeqData/Entity2PathFB_15K_237/'
    hop_num = 2
    df, rel_df = entity_to_paths(graph=graph, hop_num=hop_num, num_workers=8)
    # df = load_json_as_data_frame(entity_save_path + 'entity_to_tree_path_hop_' + str(hop_num) + ".json")
    # rel_df = triple_for_kg_train(df, hop_num=hop_num)
    # # save_to_json(data=df, file_name=entity_save_path + 'entity_to_tree_path_hop_' + str(hop_num) + ".json")
    # save_to_json(data=rel_df, file_name=entity_save_path + 'kg_entity_to_tree_path_hop_' + str(hop_num) + ".json")
    # rel_df = load_json_as_data_frame(entity_save_path + 'kg_entity_to_tree_path_hop_' + str(hop_num) + ".json")
    # for i in range(rel_df.shape[0]):
    #     print(rel_df.loc[i, 'i_num'], len(rel_df.loc[i, 'ir_path']))