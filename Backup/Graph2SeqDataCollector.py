from dgl.subgraph import DGLGraph
import torch
from time import time
import numpy as np
import math
import random
from torch import Tensor
from KGEmbedUtils.kgdataUtils import dataloader
import networkx as nx
from Backup.BallExtractor import ball_extractor
##############################################################################
################ Graph Construction over triples #############################
##############################################################################
def build_graph_from_triples_directed(num_nodes: int, triples) -> DGLGraph:
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
    g.add_edges(src, dst)
    # ===================================================================
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    g.ndata.update({'n_id': node_id})
    g.edata['e_label'] = torch.from_numpy(rel).view(-1, 1)
    # ===================================================================
    print('Constructing graph takes {:.4f} seconds'.format(time() - start))
    return g, rel
##############################################################################
#          Constructing Random walk graph on directed graph
##############################################################################
def compute_deg_norm(g: DGLGraph):
    """
    The reciprocal value of the in-degrees
    :param g:
    :return: in-degrees and out-degrees, norm_indegree and norm out_degree for random walk
    """
    np.seterr(divide='ignore', invalid='ignore')
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    out_deg = g.out_degrees(range(g.number_of_nodes())).float()
    in_deg_norm, out_deg_norm = 1.0/in_deg, 1.0/out_deg
    in_deg_norm[torch.isinf(in_deg_norm)] = 0
    out_deg_norm[torch.isinf(out_deg_norm)] = 0
    return in_deg, out_deg, in_deg_norm, out_deg_norm

def random_walk_graph(graph: DGLGraph):
    """
    Normalized the out probability distribution
    :return: rw: random walk, r_rw: reverse random walk
    """
    start = time()
    in_deg, out_deg, in_norm, out_norm = compute_deg_norm(graph)
    graph.ndata.update({'in_norm': in_norm.view(-1,1), 'out_norm': out_norm.view(-1,1)})
    graph.apply_edges(lambda edges: {'rw': edges.src['out_norm'], 'r_rw': edges.dst['in_norm']})
    print('Constructing randwalk graph in {:.4f} seconds'.format(time() - start))
    return graph, in_deg, out_deg

##############################################################################
#          Extracting balls from graph and then for random pairs generation
##############################################################################
def ball_based_path_pair_generation(graph: DGLGraph, radius=3, cut_off=None):
    def adjacent_pair_generation(ball: DGLGraph, center: int):
        ball_walker = AnonymousWalker(graph=ball)
        paths = ball_walker.all_paths_from_center(center=center, cutoff=cut_off)
        return paths
    graph_balls = ball_extractor(g=graph, radius=radius, undirected=True)
    start = time()
    for idx, ball in enumerate(graph_balls):
        if idx % 1000 == 0:
            print('{} balls have been processed in {} seconds'.format(idx, time() - start))
        ball_paths = adjacent_pair_generation(ball.ball, ball.center_map())

##############################################################################
################ Anonymous Walk on Graph         #############################
##############################################################################
class AnonymousWalker(object):
    def __init__(self, graph: DGLGraph = None):
        self.graph = graph
        self.paths = dict()
        self.nx_graph = self.graph.to_networkx()
        self.r_nx_graph = self.nx_graph.reverse()
    # =============================================================================
    def all_anonymous_paths(self, steps, keep_last=False):
        '''Get all possible anonymous walks of length up to steps.'''
        paths = []
        last_step_paths = [[0, 1]]
        for i in range(2, steps+1):
            current_step_paths = []
            for j in range(i + 1):
                for walks in last_step_paths:
                    if walks[-1] != j and j <= max(walks) + 1:
                        paths.append(walks + [j])
                        current_step_paths.append(walks + [j])
            last_step_paths = current_step_paths
        # filter only on n-steps walks
        if keep_last:
            paths = list(filter(lambda path: len(path)==steps + 1, paths))
        self.paths[steps] = paths
        return paths

    def all_anonymous_paths_edges(self, steps: int, keep_last=False):
        '''Get all possible anonymous walks of length up to steps, using edge labels'''
        paths = []
        last_step_paths = [[]]
        for i in range(0, steps):
            current_step_paths = []
            for j in range(i + 1):
                for walks in last_step_paths:
                    if j <= max(walks + [0]) + 1:
                        paths.append(walks + [j])
                        current_step_paths.append(walks + [j])
            last_step_paths = current_step_paths
        if keep_last:
            paths = last_step_paths
        self.paths[steps] = paths
        return paths
    # =============================================================================
    # =============================================================================
    def n_samples(self, steps, delta, eps):
        '''Number of samples with eps and delta concetration inequality.'''
        n = len(list(self.paths[steps]))
        estimation = 2*(math.log(2)*n + math.log(1./delta))/eps**2
        return int(estimation) + 1

    def random_step_node(self, node, out_or_not=True):
        '''Moves one step from the current according to probabilities of outgoing/incoming edges.
        Return next node.'''
        r = random.uniform(0, 1)
        low = 0
        if out_or_not: #out-edges
            successors = self.graph.successors(node)
            if successors.nelement() > 0:
                for suc_node in successors.tolist():
                    e_id = self.graph.edge_id(node, suc_node)
                    if isinstance(e_id, Tensor) and e_id.shape[0] > 1:
                        e_id = e_id[random.randint(0, e_id.shape[0]-1)]
                    p = self.graph.edata['rw'][e_id]
                    if r < low + p:
                        return (suc_node, e_id)
                    low = low + p
        else:          #in-edges
            predecessors = self.graph.predecessors(node)
            if predecessors.nelement() > 0:
                for pre_node in predecessors.tolist():
                    e_id = self.graph.edge_id(pre_node, node)
                    if isinstance(e_id, Tensor) and e_id.shape[0] > 1:
                        e_id = e_id[random.randint(0, e_id.shape[0]-1)]
                    p = self.graph.edata['r_rw'][e_id]
                    if r < low + p:
                        return (pre_node, e_id)
                    low = low + p
        return (None, None)

    def random_walk_node(self, node, steps: int, out_or_not=True):
        '''Creates anonymous walk from a node for arbitrary steps.
        Returns a tuple with consequent nodes.'''
        walk = [node]
        d = dict()
        d[node] = 0
        ano_walk = [d[node]]
        count = 1
        for i in range(steps):
            v, _ = self.random_step_node(node, out_or_not=out_or_not)
            # ==============
            if v is None:
                break
            # ==============
            if v not in d:
                d[v] = count
                count = count + 1
            walk.append(v)
            ano_walk.append(d[v])
            node = v
        return walk, ano_walk

    def random_walk_with_label_edges(self, node, steps, out_or_not=True):
        """
        Random walk with edge labels
        :param node:
        :param steps:
        :param out_or_not:
        :return:
        """
        walk = []
        idx = 0
        d = dict()
        ano_walk = []
        start_node = node
        for i in range(steps):
            v, e_id = self.random_step_node(node, out_or_not=out_or_not)
            # ==============
            if v is None: #or (v == start_node and len(walk) >= 1)
                break
            # ==============
            e_label = int(self.graph.edata['e_label'][e_id].numpy()[0])
            if e_label not in d:
                d[e_label] = idx
                idx = idx + 1
            walk.append(e_label)
            ano_walk.append(d[e_label])
            node = v
        return walk, ano_walk

    # =============================================================================
    #                 Compute all paths among two nodes
    # =============================================================================
    def all_paths_among_two_nodes(self, head, tail, cutoff=None, out_or_not=True):
        """
        Output all paths between two nodes
        :param head:
        :param tail:
        :param cutoff:
        :return:
        """
        if out_or_not:
            paths = nx.all_simple_paths(self.nx_graph, source=head, target=tail, cutoff=cutoff)
        else:
            paths = nx.all_simple_paths(self.r_nx_graph, source=head, target=tail, cutoff=cutoff)
        return paths

    def paths_to_walks_label_edges(self, paths):
        walks_pairs = []
        for path in paths:
            if len(path) > 1:
                walk, an_walk = self.entity_path_with_label_edges(path)
                walks_pairs.append((walk, an_walk))
        if len(walks_pairs) == 0:
            return None
        return walks_pairs

    def entity_path_with_label_edges(self, path):
        walk = []
        idx = 0
        d = dict()
        ano_walk = []
        for i in range(1, len(path)):
            e_id = self.graph.edge_id(path[i-1], path[i])
            if isinstance(e_id, Tensor) and e_id.shape[0] > 1:
                e_id = e_id[random.randint(0, e_id.shape[0] - 1)]
            e_label = int(self.graph.edata['e_label'][e_id].numpy()[0])
            if e_label not in d:
                d[e_label] = idx
                idx = idx + 1
            walk.append(e_label)
            ano_walk.append(d[e_label])
        return walk, ano_walk

    def all_paths_center(self, center, cutoff=None, out_or_not=True):
        path_num = 0
        dict_paths = dict()
        for node_id in range(self.graph.number_of_nodes()):
            if node_id != center:
                paths = list(self.all_paths_among_two_nodes(center, node_id, cutoff=cutoff, out_or_not=out_or_not))
                if len(paths) > 0:
                    dict_paths[node_id] = paths
                path_num = path_num + len(paths)
        return dict_paths, path_num

    # =============================================================================
    #                 Compute all shortest paths from/to a given node
    # =============================================================================
    def shortest_path_center(self, center, cutoff=None, out_or_not=True):
        """
        All the shortest paths from a given center
        :param center:
        :param cutoff:
        :return:
        """
        if out_or_not:
            dict_paths = nx.single_source_shortest_path(self.nx_graph, source=center, cutoff=cutoff)
        else:
            dict_paths = nx.single_source_shortest_path(self.r_nx_graph, source=center, cutoff=cutoff)
        return dict_paths
##############################################################################
#############End of Anonymous Walk on Graph         #############################
#############################################################################

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def edge_random_walk(awer: AnonymousWalker, num_of_walks, steps, weights, node_list, out_or_not=True):
    """
    Random walk along the graph and generate (num_of_walks) relation type sequences
    :param awer:
    :param num_of_walks:
    :param steps:
    :param weights:
    :param node_list:
    :param out_or_not:
    :return:
    """
    from numpy.random import choice
    def random_walk_with_edge_label(node_id):
        walk_tuple, ano_walk_tuple = awer.random_walk_with_label_edges(node_id, steps=steps, out_or_not=out_or_not)
        if len(walk_tuple) < steps:#Padding -1 to make sure all random walks have the same length
            padding = [-1]*(steps - len(walk_tuple))
            walk_tuple, ano_walk_tuple = walk_tuple + padding, ano_walk_tuple + padding
        return node_id, walk_tuple, ano_walk_tuple
    nodes, rel_rw_list, arw_list = [], [], []
    start = time()
    for idx, n_id in enumerate(choice(node_list, num_of_walks, p=weights)):
        if idx % 2000==0:
            print('{} random walks have been completed in {:.2f} seconds'.format(idx, time() - start))
        rw_result = random_walk_with_edge_label(n_id)
        nodes.append(rw_result[0])
        rel_rw_list.append(rw_result[1])
        arw_list.append(rw_result[2])
    return nodes, rel_rw_list, arw_list
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def adjacent_sequence_pair_collection(awer: AnonymousWalker, num_of_pairs, out_weights, in_weights,
                                      node_list, num_rels, cut_off=None, out_or_not=True):
    """
    :param awer:
    :param num_of_pairs:
    :param weights:
    :param node_list:
    :param cut_off:
    :param out_or_not:
    :return:
    """
    from numpy.random import choice
    def paths_among_two_nodes(from_node, to_node):
        paths = awer.all_paths_among_two_nodes(head=from_node, tail=to_node, cutoff=cut_off)
        walk_edge_pairs = awer.paths_to_walks_label_edges(paths)
        return walk_edge_pairs

    from_node_list = choice(node_list, num_of_walks, p=out_weights)
    to_node_list = choice(node_list, num_of_walks, p=in_weights)
    start = time()
    sequence_number = 0

    counts = np.zeros(num_rels)
    for idx in range(num_of_pairs):
        if idx % 2000==0:
            print('{} random walks have been completed in {:.2f} seconds'.format(idx, time() - start))
        from_node_i, to_node_i = from_node_list[idx], to_node_list[idx]
        walks_i = paths_among_two_nodes(from_node=from_node_i, to_node=to_node_i)
        if walks_i is not None:
            for walk in walks_i:
                for en in walk:
                    counts[en] = counts[en] + 1

        if walks_i is not None:
            sequence_number = sequence_number + len(walks_i)

        print(idx, sequence_number)

    print(counts)


###==================================================================================================
###                           WordNet Data Generation
###==================================================================================================
def WordNetRWDataCollection(num_of_walks: int, steps: int, random_seed, save_folder=None, out_or_not=True):
    data = dataloader('WN18RR')  # There are 7 edges which connect same entities
    num_nodes = data.num_entities
    train_triples, val_triples, test_triples = data.train.transpose(), data.valid.transpose(), \
                                               data.test.transpose()
    graph, rel = build_graph_from_triples_directed(num_nodes=num_nodes, triples=train_triples)
    start = time()
    rw_graph, in_degree, out_degree = random_walk_graph(graph=graph)
    if out_or_not:
        sample_weights = (out_degree/torch.sum(out_degree)).numpy()
    else:
        sample_weights = (in_degree/torch.sum(in_degree)).numpy()
    node_list = np.arange(0, num_nodes).tolist()
    awer = AnonymousWalker(rw_graph)
    randow_walks = edge_random_walk(awer=awer, num_of_walks=num_of_walks, steps=steps,
                                    weights=sample_weights, node_list=node_list, out_or_not=out_or_not)
    print('Random walking takes {} seconds'.format(time() - start))
    random_walk_file_name = 'rand_walk_'+str(num_of_walks) + '_' + str(steps) + 'rand_' + str(random_seed)
    random_walk_saver(save_folder, random_walk_file_name, randow_walks)
    return randow_walks

def WordNetRWPairCollection(num_of_pairs: int, random_seed, radius=3, cut_off=None, out_or_not=True, save_folder=None):
    """
    :param num_of_pairs:
    :param random_seed:
    :param radius:
    :param cut_off:
    :param out_or_not:
    :param save_folder:
    :return:
    """
    data = dataloader('WN18RR')  # There are 7 edges which connect same entities
    num_relation = data.num_relation
    num_nodes = data.num_entities
    train_triples, val_triples, test_triples = data.train.transpose(), data.valid.transpose(), \
                                               data.test.transpose()
    graph, rel = build_graph_from_triples_directed(num_nodes=num_nodes, triples=train_triples)
    start = time()
    rw_graph, in_degree, out_degree = random_walk_graph(graph=graph)
    if out_or_not:
        out_sample_weights = (out_degree/torch.sum(out_degree)).numpy()
        in_sample_weights = (in_degree/torch.sum(in_degree)).numpy()
    else:
        out_sample_weights = (in_degree/torch.sum(in_degree)).numpy()
        in_sample_weights = (out_degree/torch.sum(out_degree)).numpy()
    node_list = np.arange(0, num_nodes).tolist()
    awer = AnonymousWalker(rw_graph) # Anonymous random walker

    adjacent_sequence_pair_collection(awer=awer, num_of_pairs=num_of_pairs, out_weights=out_sample_weights,
                                      in_weights=in_sample_weights, node_list=node_list,
                                      num_rels=num_relation, cut_off=cut_off,
                                      out_or_not=out_or_not)
    print('Sequence pair generation takes {} seconds'.format(time() - start))
    return None

###==================================================================================================
###                           FreeBase Data Generation
###==================================================================================================
def FreeBaseRWDataCollection(num_of_walks: int, steps: int, random_seed, save_folder=None, out_or_not=True):
    data = dataloader('FB15k-237')  # There are 1625 edges which connect same entities
    num_nodes = data.num_entities
    train_triples, val_triples, test_triples = data.train.transpose(), data.valid.transpose(), data.test.transpose()
    graph, rel = build_graph_from_triples_directed(num_nodes=num_nodes, triples=train_triples)
    start = time()
    rw_graph, in_degree, out_degree = random_walk_graph(graph=graph)
    if out_or_not:
        sample_weights = (out_degree/torch.sum(out_degree)).numpy()
    else:
        sample_weights = (in_degree/torch.sum(in_degree)).numpy()
    node_list = np.arange(0, num_nodes).tolist()
    awer = AnonymousWalker(rw_graph)
    randow_walks = edge_random_walk(awer=awer, num_of_walks=num_of_walks,
                                    steps=steps, weights=sample_weights,
                                    node_list=node_list, out_or_not=out_or_not)
    print('Random walking takes {} seconds'.format(time() - start))
    random_walk_file_name = 'rand_walk_'+str(num_of_walks) + '_' + str(steps) + 'rand_' + str(random_seed)
    random_walk_saver(save_folder, random_walk_file_name, randow_walks)
    return randow_walks


def FreeBaseRWPairCollection(num_of_pairs: int, random_seed, radius=3, cut_off=None, out_or_not=True, save_folder=None):
    """
    :param num_of_pairs:
    :param random_seed:
    :param radius:
    :param cut_off:
    :param out_or_not:
    :param save_folder:
    :return:
    """
    data = dataloader('FB15k-237')  # There are 1625 edges which connect same entities
    num_relation = data.num_relation
    num_nodes = data.num_entities
    train_triples, val_triples, test_triples = data.train.transpose(), data.valid.transpose(), \
                                               data.test.transpose()
    graph, rel = build_graph_from_triples_directed(num_nodes=num_nodes, triples=train_triples)
    start = time()
    rw_graph, in_degree, out_degree = random_walk_graph(graph=graph)
    if out_or_not:
        out_sample_weights = (out_degree/torch.sum(out_degree)).numpy()
        in_sample_weights = (in_degree/torch.sum(in_degree)).numpy()
    else:
        out_sample_weights = (in_degree/torch.sum(in_degree)).numpy()
        in_sample_weights = (out_degree/torch.sum(out_degree)).numpy()
    node_list = np.arange(0, num_nodes).tolist()
    awer = AnonymousWalker(rw_graph) # Anonymous random walker

    adjacent_sequence_pair_collection(awer=awer, num_of_pairs=num_of_pairs, out_weights=out_sample_weights,
                                      in_weights=in_sample_weights, node_list=node_list, cut_off=cut_off,
                                      num_rels=num_relation,
                                      out_or_not=out_or_not)
    print('Sequence pair generation takes {} seconds'.format(time() - start))
    return None

##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##  Data saver and loader
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def random_walk_saver(folder_name, file_name, random_walks):
    relation_rw_file_name = folder_name + file_name + '_rel'
    anonymous_rw_file_name = folder_name + file_name + '_aw'
    anonymous_node_file_name = folder_name + file_name + '_node'
    node_array = np.asarray(random_walks[0], dtype=np.int32)
    rel_rw_array = np.asarray(random_walks[1], dtype=np.int16)
    a_rw_array = np.asarray(random_walks[2], dtype=np.int16)

    np.savez(relation_rw_file_name, rel_rw_array=rel_rw_array)
    np.savez(anonymous_rw_file_name, a_rw_array=a_rw_array)
    np.savez(anonymous_node_file_name, node_array=node_array)

def random_walk_loader(folder_name, num_of_walks: int, steps: int, random_seed):
    file_name = 'rand_walk_' + str(num_of_walks) + '_' + str(steps) + 'rand_' + str(random_seed)
    relation_rw_file_name = folder_name + file_name + '_rel.npz'
    anonymous_rw_file_name = folder_name + file_name + '_aw.npz'
    anonymous_node_file_name = folder_name + file_name + '_node.npz'

    rel_rw_array = np.load(relation_rw_file_name)
    a_rw_array = np.load(anonymous_rw_file_name)
    node_array = np.load(anonymous_node_file_name)
    return node_array['node_array'], rel_rw_array['rel_rw_array'], a_rw_array['a_rw_array']
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == '__main__':

    WN_Folder_Name = '../SeqData/RandWalk_WN18RR/'
    FB_Folder_Name = '../SeqData/AWK_FB15K_237/'
    #======================================================================
    random_seed = 2019
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    num_of_walks = 1000000
    steps = 20
    #======================================================================
    cut_off = 8
    num_of_pairs = 100000
    # ======================================================================
    # node_array, rel_rw_array, a_rw_array = random_walk_loader(WN_Folder_Name,
    #                                                           num_of_walks=num_of_walks, steps=steps, random_seed=random_seed)
    # unique_elements, counts_elements = np.unique(node_array, return_counts=True)
    # print(unique_elements, counts_elements)
    # WordNetRWDataCollection(num_of_walks=num_of_walks, steps=steps, random_seed=random_seed, save_folder=WN_Folder_Name)
    # ======================================================================
    # FreeBaseRWDataCollection(num_of_walks=num_of_walks, steps=steps, random_seed=random_seed, save_folder=FB_Folder_Name)
    # node_array, rel_rw_array, a_rw_array = random_walk_loader(FB_Folder_Name,
    #                                                           num_of_walks=num_of_walks, steps=steps, random_seed=random_seed)
    # unique_elements, counts_elements = np.unique(node_array, return_counts=True)
    # print(unique_elements.shape, counts_elements.max())
    # ======================================================================
    WordNetRWPairCollection(num_of_pairs=num_of_pairs, random_seed=random_seed, cut_off=cut_off, save_folder=None)
    # FreeBaseRWPairCollection(num_of_pairs=num_of_pairs, random_seed=random_seed, cut_off=cut_off, save_folder=None)
