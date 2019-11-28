data_path = '../KGData/'
import numpy as np

from dgl.subgraph import DGLGraph, DGLSubGraph
import numpy as np
import torch
from torch import Tensor
from time import time
from networkx.algorithms.distance_measures import diameter, radius

def dict_collection(kg_name, data_file_names):
    kg_data_path = data_path + kg_name + '/'
    rel_dict = dict()
    ent_dict = dict()
    data_dict = dict()
    rel_idx = 0
    ent_idx = 0
    row_counts = []
    for f_idx, file_name in enumerate(data_file_names):
        file_name_i = kg_data_path + file_name + '.txt'
        data = []
        with open(file_name_i, mode='r') as fp:
            line = fp.readline()
            cnt = 1
            while line:
                tokens = line.strip().split("\t")
                head, relation, tail = tokens[0].strip(), tokens[1].strip(), tokens[2].strip()
                if relation not in rel_dict:
                    rel_dict[relation] = rel_idx
                    rel_idx = rel_idx + 1
                if head not in ent_dict:
                    ent_dict[head] = ent_idx
                    ent_idx = ent_idx + 1
                if tail not in ent_dict:
                    ent_dict[tail] = ent_idx
                    ent_idx = ent_idx + 1
                data.append([ent_dict[head], rel_dict[relation], ent_dict[tail]])
                line = fp.readline()
                cnt = cnt + 1
        row_counts.append(cnt)
        data_dict[file_name] = np.array(data)
    return data_dict, ent_dict, rel_dict

class dataloader:
    def __init__(self, kg_file_name:str):
        self.file_names = ['train', 'valid', 'test']
        if kg_file_name == 'WN18':
            self.kg_file_name = kg_file_name + '/text/'
        elif kg_file_name == 'WN18RR':
            self.kg_file_name = kg_file_name + '/original/'
        else:
            self.kg_file_name = kg_file_name
        data_dict, self.ent_dict, self.rel_dict = dict_collection(kg_name=self.kg_file_name,
                                                                  data_file_names=self.file_names)
        self.train = data_dict['train']
        self.valid = data_dict['valid']
        self.test = data_dict['test']
        self.num_relation = len(self.rel_dict)
        self.num_entities = len(self.ent_dict)
        self.rel_dict_rev = {v: k for k, v in self.rel_dict.items()}

def compute_deg(g: DGLGraph):
    """
    The reciprocal value of the in-degrees
    :param g:
    :return:
    """
    np.seterr(divide='ignore', invalid='ignore')
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    out_deg = g.out_degrees(range(g.number_of_nodes())).float()
    deg_sum = in_deg + out_deg
    print('Single Node', (deg_sum == 0).sum())
    return in_deg, out_deg

def build_graph_from_triples_bi_direction(num_nodes: int, triples, multi_graph=False) -> DGLGraph:
    """
    :param num_nodes:
    :param num_relations:
    :param triples: 3 x number of edges
    :return:
    """
    start = time()
    rev_triples = triples[[2,1,0]]
    g, rel, in_deg, out_deg, multi_edge_nodes = build_graph_from_triples_directed(num_nodes=num_nodes, triples=triples)
    r_g, r_rel, r_in_deg, r_out_deg, r_multi_edge_nodes = build_graph_from_triples_directed(num_nodes=num_nodes, triples=rev_triples)
    print((rel - r_rel).sum(), (in_deg - r_out_deg).sum(), (out_deg - r_in_deg).sum())
    print('Constructing graph takes {:.2f} seconds'.format(time() - start))
    return g, r_g


def build_graph_from_triples_directed(num_nodes: int, triples, multi_graph=False) -> DGLGraph:
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
    in_deg, out_deg = compute_deg(g)
    rel = torch.from_numpy(rel)
    g.ndata.update({'n_id': node_id, 'in_degree': in_deg, 'out_degree': out_deg})
    g.edata['e_label'] = rel
    # ===================================================================
    g.apply_edges(lambda edges: {'node_id_pair': torch.cat((edges.src['n_id'], edges.dst['n_id']), dim=-1)})
    multi_edge_nodes = multi_edge_node_pairs(g)
    # ===================================================================
    print('Constructing graph takes {:.2f} seconds'.format(time() - start))

    return g, rel, in_deg, out_deg, multi_edge_nodes

def multi_edge_node_pairs(graph: DGLGraph):
    """
    There are pairs of entities between which there are multiple edges/relations
    :param graph:
    :return:
    """
    multi_edge_node_ids = []
    node_id_pairs = graph.edata['node_id_pair'].tolist()
    graph.edata['m_edge'] = torch.zeros(graph.number_of_edges()).type(torch.ByteTensor)
    graph.edata['m_edge_id'] = torch.zeros((graph.number_of_edges(), 2), dtype=torch.long)
    node_pair_set = set()
    for idx, pair in enumerate(node_id_pairs):
        str_pair = ' '.join([str(x) for x in pair])
        if str_pair in node_pair_set:
            if (pair[0], pair[1]) not in multi_edge_node_ids:
                multi_edge_node_ids.append((pair[0], pair[1]))
        else:
            node_pair_set.add(str_pair)
    ante_set = set([x[0] for x in multi_edge_node_ids])
    cons_set = set([x[1] for x in multi_edge_node_ids])
    multi_edge_node_ids = set(multi_edge_node_ids)
    for idx, pair in enumerate(multi_edge_node_ids):
        from_id, to_id = pair[0], pair[1]
        e_ids = graph.edge_ids(from_id, to_id)
        graph.edata['m_edge'][e_ids] = 1
        e_id_pair = torch.from_numpy(np.array([(idx + 1), e_ids.shape[0]], dtype=np.int64))
        graph.edata['m_edge_id'][e_ids] = e_id_pair
    return multi_edge_node_ids, ante_set, cons_set


#==============================================================================================
def graph_statistics(graph: DGLGraph):
    network_g = graph.to_networkx()
    d, r = diameter(network_g), radius(network_g)
    return {'diameter': d, 'radius': r}

if __name__ == '__main__':
    data = dataloader('FB15K-237')
    # print(data.num_relation, data.num_entities, data.train.shape, data.valid.shape, data.test.shape)
    # print(data.train)
    #
    # build_graph_from_triples_bi_direction(data.num_entities, data.train.transpose())
    # file_names = ['train', 'valid', 'test']
    # a, b, c = dict_collection('FB15K-237', file_names)
    # print(len(c))
    # with open(file_name, mode='r') as fp:
    #     line = fp.readline()
    #     cnt = 1
    #     while line:
    #         # print("Line {}: {}".format(cnt, line.strip()))
    #         print(len(line.split('\t')))
    #         line = fp.readline()
    #         cnt += 1
    # print(cnt)