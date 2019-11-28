from time import time
from dgl.contrib.sampling import NeighborSampler
from dgl.nodeflow import NodeFlow
from pandas import DataFrame
import pandas as pd
from Backup.KGUtils import build_graph_from_triples_directed
from dgl import DGLGraph
import numpy as np
from KGEmbedUtils.kgdataUtils import dataloader
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)


def tree_extractor(g: DGLGraph, radius: int, num_workers=8):
    """
    :param g: whole graph
    :param radius: radius of each ball (bi-directional)
    :return: a set of sub-graphs (each ball is extracted for each node)
    """
    g.readonly(readonly_state=True)
    def NodeFlow2Tree(nf: NodeFlow):
        nodes_list = []
        center = nf.layer_parent_nid(-1)[0]
        for i in range(0, radius):
            nodes_list.append((i + 1, set(nf.layer_parent_nid(radius - i - 1).tolist())))
        return center, nodes_list
    expand_factor = g.number_of_nodes()
    out_tree_list = []
    for nf_out in NeighborSampler(g=g, expand_factor=expand_factor, batch_size=1, neighbor_type='out',
                              shuffle=False, num_hops=radius, num_workers=num_workers):
        center, out_nodes = NodeFlow2Tree(nf_out)
        out_tree_list.append((center.item(), out_nodes))
        # print('out', center, out_nodes)
    out_tree = pd.DataFrame(out_tree_list, columns=['center', 'out_nodes'])
    in_tree_list = []
    for nf_in in NeighborSampler(g=g, expand_factor=expand_factor, batch_size=1, neighbor_type='in',
                                 shuffle=False, num_hops=radius, num_workers=num_workers):
        center, int_nodes = NodeFlow2Tree(nf_in)
        # print('in', center.item(), int_nodes)
        in_tree_list.append((center.item(), int_nodes))
    in_tree = pd.DataFrame(in_tree_list, columns=['center', 'in_nodes'])

    tree = pd.merge(left=out_tree, right=in_tree, on='center', how='inner')
    g.readonly(readonly_state=False)
    return tree

def tree_to_shortest_path(tree_df: DataFrame, graph: DGLGraph):
    """
    :param tree_df: center, in_nodes, out_nodes
    :return:
    """
    def path_row(row):
        center, in_nodes, out_nodes = row['center'], row['in_nodes'], row['out_nodes']
        # print(len(in_nodes), len(out_nodes))
        def shortest_path(center, red_node_list, neighbor_type: str):
            if len(red_node_list[0][1]) == 0:
                return [], set()
            shortest_path_list = [[] for _ in range(len(red_node_list))]
            # shortest_path_list[0] = [(center, x) for x in red_node_list[0][1]]
            leaf_node_dict = dict()
            for x in red_node_list[0][1]:
                shortest_path_list[0].append((center, x))
                leaf_node_dict[x] = 0

            if len(red_node_list) > 1:
                for i in range(1, len(red_node_list)):
                    for entity in red_node_list[i][1]:
                        leaf_node_dict[entity] = 0
                        for pre_path in shortest_path_list[i-1]:
                            pre_entity = pre_path[-1]
                            if neighbor_type == 'out':
                                if graph.has_edge_between(pre_entity, entity):
                                    shortest_path_list[i].append(pre_path + (entity,))
                                    leaf_node_dict[pre_entity] = leaf_node_dict[pre_entity] + 1
                                    break
                                else:
                                    continue
                            else:
                                if graph.has_edge_between(entity, pre_entity):
                                    shortest_path_list[i].append(pre_path + (entity,))
                                    leaf_node_dict[pre_entity] = leaf_node_dict[pre_entity] + 1
                                    break
                                else:
                                    continue
            leaf_node_set = set([key for key, value in leaf_node_dict.items() if value == 0])
            return shortest_path_list, leaf_node_set

        def reduce_node(nodes_list):
            union_set = nodes_list[0][1]
            red_node_list = [nodes_list[0]]
            if len(nodes_list[0][1]) == 0:
                return red_node_list
            for i in range(1, len(nodes_list)):
                if len(nodes_list[i][1]) == 0:
                    return red_node_list
                diff_set = (nodes_list[i][1]).difference(union_set)
                if len(diff_set) > 0:
                    red_node_list.append((nodes_list[i][0], diff_set))
                    union_set = union_set.union(diff_set)
            return red_node_list
        red_in_nodes = reduce_node(in_nodes)
        red_out_nodes = reduce_node(out_nodes)

        shortest_in_paths, in_leaf_node_set = shortest_path(center, red_in_nodes, neighbor_type='in')
        shortest_out_paths, out_leaf_node_set = shortest_path(center, red_out_nodes, neighbor_type='out')
        # print('In {} degree {} {} ===> {}'.format(center, graph.in_degree(center), shortest_in_paths, red_in_nodes))
        # print('Out {} degree {} {} ===> {}'.format(center, graph.out_degree(center), shortest_out_paths, red_out_nodes))
        return shortest_in_paths, shortest_out_paths, in_leaf_node_set, out_leaf_node_set
    tree_df[['in_path', 'out_path', 'in_leaf', 'out_leaf']] = tree_df.parallel_apply(path_row, axis=1, result_type="expand")
    return tree_df

def entity_path_to_relation(data: DataFrame, graph: DGLGraph, walk_len: int):
    # center, out_nodes, in_nodes, in_path, out_path
    def entity_path_padding(row):
        in_paths, out_paths = row['in_path'], row['out_path']
        in_leaf_set, out_leaf_set = row['in_leaf'], row['out_leaf']
        def path_padding(paths, neighbor_type: str, leaf_set: set):
            if len(paths) == 0:
                path_count, leaf_num = 0, 0
            else:
                path_num_list = [len(x) for x in paths]
                path_count, leaf_num = sum(path_num_list), len(leaf_set)
            if path_count == 0:
                return np.full((path_count, walk_len + 3), fill_value=-1, dtype=np.int64), 0, 0
            path_array = np.full((path_count, walk_len + 3), fill_value=-1, dtype=np.int64)
            idx = 0
            for path_list in paths:
                for path_i in path_list:
                    path_array[idx, -2] = (path_i[-1] in leaf_set)
                    if neighbor_type == 'in':
                        path_i = tuple(reversed(path_i))
                        path_array[idx, -1] = 0
                    path_array[idx, :len(path_i)] = path_i
                    path_array[idx, -3] = len(path_i)
                    idx = idx + 1
            return path_array, path_count, leaf_num
        in_path_array, in_path_num, in_leaf_path_num = path_padding(paths=in_paths, neighbor_type='in', leaf_set=in_leaf_set)
        out_path_array,out_path_num, out_leaf_path_num = path_padding(paths=out_paths, neighbor_type='out', leaf_set=out_leaf_set)
        return in_path_array, out_path_array, in_path_num, out_path_num, in_leaf_path_num, out_leaf_path_num
    data[['in_path_pad', 'out_path_pad', 'ip_cnt', 'op_cnt', 'ile_n', 'ole_n']] = data.parallel_apply(entity_path_padding, axis=1, result_type='expand')
    return data

def entity_to_paths(graph: DGLGraph, radius: int, num_workers=8):
    start = time()
    print('Step 1: Tree extracting...')
    tree_df = tree_extractor(g=graph, radius=radius, num_workers=num_workers)
    print('\tTree generation of {} nodes in {:.2f} seconds'.format(graph.number_of_nodes(), time() - start))
    print('Step 2: Shortest path extracting...')
    start = time()
    entity_path_df = tree_to_shortest_path(tree_df=tree_df, graph=graph)
    print('\tShortest generation of {} nodes in {:.2f} seconds'.format(graph.number_of_nodes(), time() - start))
    print('Step3: Entity path to relation path...')
    entity_path_df = entity_path_to_relation(data=entity_path_df, graph=graph, walk_len=radius + 1)
    print('\tEntity path transformation in {:.2f} seconds'.format(time() - start))
    drop_names = [col for col in entity_path_df.columns if col not in {'center', 'in_path_pad', 'out_path_pad', 'ip_cnt', 'op_cnt', 'ile_n', 'ole_n'}]
    entity_path_df = entity_path_df.drop(columns=drop_names)
    return entity_path_df


if __name__ == '__main__':
    path = '../Data/EntityEmbeder/'
    data = dataloader('FB15k-237')
    # data = dataloader('WN18RR')
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
    # print(in_degree[0], out_degree[0])
    df = entity_to_paths(graph=graph, radius=3, num_workers=8)
    print(df[['ip_cnt', 'ile_n', 'op_cnt', 'ole_n']].sum())
    # print(df.shape)
    # for col in df.columns:
    #     print(col)

    print(df.head(1).values)