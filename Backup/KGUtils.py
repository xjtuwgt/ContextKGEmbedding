from dgl.subgraph import DGLGraph, DGLSubGraph
import numpy as np
import torch
from torch import Tensor
from time import time
from networkx.algorithms.distance_measures import diameter, radius

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

def build_graph_from_triples(num_nodes: int, num_relations: int, triples) -> DGLGraph:
    """
    :param num_nodes:
    :param num_relations:
    :param triples: 3 x number of edges
    :return:
    """
    start = time()
    g = DGLGraph()
    g.add_nodes(num_nodes)
    src, rel, dst = triples
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src)) #Bi-direction graphs
    rel = np.concatenate((rel, rel + num_relations)) #reverse relation = relation + number of relations
    edges = sorted(zip(dst, src, rel))
    dst, src, rel = np.array(edges).transpose()
    g.add_edges(src, dst)

    # ===================================================================
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    in_deg, out_deg = compute_deg(g)
    rel = torch.from_numpy(rel)
    g.ndata.update({'n_id': node_id, 'in_degree': in_deg, 'out_degree': out_deg})
    g.edata['e_label'] = rel
    # ===================================================================
    g.apply_edges(lambda edges: {'node_id_pair': torch.cat((edges.src['n_id'], edges.dst['n_id']), dim=-1)})
    # ===================================================================
    print('Constructing graph takes {:.2f} seconds'.format(time() - start))
    return g, rel, in_deg, out_deg


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

##===============================================================================================
def Graph2LineGraph(g: DGLGraph):
    """
    :param g: original graph: 'n_id', 'e_id', 'node_id_pair' where node_id_pair can be viewed as the identifier
    :return:
    """
    start = time()
    line_g = g.line_graph()
    line_g.ndata['n_id'] = g.edata['e_label'].unsqueeze(-1)
    line_g.ndata['edge_ids'] = g.edata['node_id_pair']
    line_g.apply_edges(lambda edges: {'e_label': edges.src['edge_ids'][:,1]})
    line_g.ndata.pop('edge_ids')
    # ===================================================================
    line_g.apply_edges(lambda edges: {'node_id_pair': torch.cat((edges.src['n_id'], edges.dst['n_id']), dim=-1)})
    # ===================================================================
    print('Converting to line-graph takes {:.2f} seconds'.format(time() - start))
    return line_g

def Ball2LineGraph(ball: DGLSubGraph):
    if ball.is_readonly:
        ball.readonly(readonly_state=False)
    ball.apply_edges(lambda edges: {'edge_ids': torch.cat((edges.src['n_id'].unsqueeze(-1), edges.dst['n_id'].unsqueeze(-1)), -1)})
    ball_line = ball.line_graph()
    ball_line.ndata['n_id'] = ball.edata['type']
    ball_line.ndata['edge_ids'] = ball.edata.pop('edge_ids')
    ball_line.apply_edges(lambda edges: {'e_id': edges.src['edge_ids'][:, 1]})
    ball_line.ndata.pop('edge_ids')
    return ball_line
##===============================================================================================
def get_adj_and_degrees(num_nodes, triplets):
    """
    Get the adjacency list and degrees of the graph
    :param num_nodes:
    :param triples:
    :return:
    """
    adj_list = [[] for _ in range(num_nodes)]
    for i,triplet in enumerate(triplets):
        adj_list[triplet[0]].append([i, triplet[2]])
        adj_list[triplet[2]].append([i, triplet[0]])

    degrees = np.array([len(a) for a in adj_list])
    adj_list = [np.array(a) for a in adj_list]
    return adj_list, degrees

def graph2triplets(graph: DGLGraph):
    graph.apply_edges(lambda edges: {'triple': torch.cat((edges.src['n_id'],
                                                          edges.data['e_label'].unsqueeze(-1),
                                                          edges.dst['n_id']), dim=-1)})
    return graph.edata.pop('triple').numpy()