from dgl.subgraph import DGLGraph, DGLSubGraph
import torch
from torch import Tensor
from time import time
from dgl.contrib.sampling import NeighborSampler
from dgl.nodeflow import NodeFlow
import numpy as np

##############################################################################
#          Constructing Random walk graph on directed graph
##############################################################################
def compute_deg_norm(g: DGLGraph):
    """
    The reciprocal value of the in-degrees
    :param g:
    :return:
    """
    np.seterr(divide='ignore', invalid='ignore')
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    out_deg = g.out_degrees(range(g.number_of_nodes())).float()
    in_deg_norm, out_deg_norm = 1.0/in_deg, 1.0/out_deg
    in_deg_norm[torch.isinf(in_deg_norm)] = 0
    out_deg_norm[torch.isinf(out_deg_norm)] = 0
    return in_deg_norm, out_deg_norm

def random_walk_graph(graph: DGLGraph):
    """
    Normalized the out probability distribution
    :return:
    """
    in_norm, out_norm = compute_deg_norm(graph)
    graph.ndata.update({'in_norm': in_norm.view(-1,1), 'out_norm': out_norm.view(-1,1)})
    graph.apply_edges(lambda edges: {'rw': edges.src['out_norm'], 'r_rw': edges.dst['in_norm']})
    return graph
##############################################################################
################ Ball Extractor from a large graph############################
################ If the number of edges of the ball is zero, this means there
################ are no related triples over these entities
##############################################################################
class Ball(object):
    def __init__(self, center=-1, radius=-1, nodes=None):
        self.center = center
        self.radius = radius
        self.nodes = nodes
        if nodes is None:
            self.num_nodes = 0
        else:
            self.num_nodes = len(nodes)
        self.ball: DGLGraph = None
        self.n_dic = None

    def ball_graph(self, g: DGLGraph) -> DGLSubGraph:
        ball = g.subgraph(self.nodes)
        sub_nodes = ball.nodes().tolist()
        parent_nodes = ball.parent_nid.tolist()
        self.n_dic = dict(zip(parent_nodes, sub_nodes))
        ball.ndata['n_id'] = ball.parent_nid.view(-1,1)
        ball.edata['e_label'] = g.edata['e_label'][ball.parent_eid]
        return ball

    def center_map(self):
        if isinstance(self.center, Tensor):
            return self.n_dic[int(self.center.item())]
        else:
            return self.n_dic[int(self.center)]

    def set_radius(self, radius):
        self.radius = radius

    def set_graph(self, graph: DGLGraph):
        self.ball = graph
        if self.nodes is None:
            self.nodes = graph.ndata['n_id'].squeeze(dim=-1).tolist()
            self.num_nodes = graph.number_of_nodes()

    def set_center(self, center):
        self.center = center

    def set_node_dict(self, n_dict):
        self.n_dic = n_dict


def ball_extractor(g: DGLGraph, radius: int, undirected=False, count_limit=None):
    """
    :param g: whole graph
    :param radius: radius of each ball (bi-directional)
    :return: a set of sub-graphs (each ball is extracted for each node)
    """
    g.readonly(readonly_state=True)
    expand_factor = g.number_of_nodes()
    def NodeFlow2Ball(nf: NodeFlow):
        center = nf.layer_parent_nid(-1)[0]
        if undirected:
            nf_off = NeighborSampler(g=g, expand_factor=expand_factor, batch_size=1, neighbor_type='in',
                              shuffle=False, num_hops=radius, seed_nodes=center).fetch(0)[0]
        nodes = [center.item()]
        for i in range(0, radius):
            nodes = nodes + nf.layer_parent_nid(i).tolist()
            if undirected:
                nodes = nodes + nf_off.layer_parent_nid(i).tolist()
        nodes = list(set(nodes))
        ball = Ball(center, radius, nodes)
        ball.ball = ball.ball_graph(g)
        if ball.ball.number_of_edges() > 0:
            rw_ball_graph = random_walk_graph(ball.ball)
            ball.set_graph(rw_ball_graph)
        return ball

    expand_factor = g.number_of_nodes()
    ball_graphs = []
    idx = 0
    start = time()
    node_num_mean = 0.0
    edge_num_mean = 0.0
    ball_count = 0
    for nf in NeighborSampler(g=g, expand_factor=expand_factor, batch_size=1, neighbor_type='out',
                              shuffle=False, num_hops=radius):
        ball_i = NodeFlow2Ball(nf)
        # ===========================================================================================
        if ball_i.ball.number_of_edges() > 0:
            ball_count = ball_count + 1
            node_num_mean = node_num_mean + ball_i.num_nodes
            edge_num_mean = edge_num_mean + ball_i.ball.number_of_edges()
            ball_graphs.append(ball_i)
        # ===========================================================================================
        idx = idx + 1
        if idx % 1000 == 0:
            print('{} sub-graphs are generated in {:.2f} seconds'.format(idx, time()-start))
        if count_limit is not None and ball_count >= count_limit:
            print('{} sub-graphs are generated in {:.2f} seconds'.format(ball_count, time() - start))
            print('Average ratio of nodes in sub-graph = {:.3f}%, average number of nodes = {:.2f}/{}, '
                  'average number of edges = {:.2f}'
                  .format((node_num_mean / ball_count) *100.0/g.number_of_nodes(), node_num_mean/ball_count, g.number_of_nodes(),
                  edge_num_mean/ball_count))
            return ball_graphs
    print('{} sub-graphs are generated in {:.2f} seconds'.format(ball_count, time() - start))
    print('Average ratio of nodes in sub-graph = {:.3f}%, average number of nodes = {:.2f}/{}, '
          'average number of edges = {:.2f}'
          .format((node_num_mean / ball_count) * 100.0 / g.number_of_nodes(), node_num_mean/ball_count, g.number_of_nodes(),
                  edge_num_mean / ball_count))
    g.readonly(readonly_state=False)
    return ball_graphs