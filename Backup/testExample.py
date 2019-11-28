import pandas as pd
import numpy as np
import random, bisect
from scipy import sparse

import dgl
import torch
from RelPathCollector.Entity2TreePath import tree_extractor, tree_to_shortest_path, entity_path_to_relation
from Backup.KGUtils import multi_edge_node_pairs

def build_karate_club_graph():
    g = dgl.DGLGraph()
    node_id = torch.arange(0, 20, dtype=torch.long).view(-1, 1)
    # add 34 nodes into the graph; nodes are labeled from 0~33
    g.add_nodes(20)
    g.ndata['n_id'] = node_id
    # all 78 edges as a list of tuples
    edge_list = [(0,1), (2, 0), (2, 1), (3, 0), (2,1), (2,1)]
    # add edges two lists of nodes: src and dst
    # g.add_edges(src, dst)
    # g.add_nodes(34)
    # # all 78 edges as a list of tuples
    # edge_list = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
    #     (4, 0), (5, 0), (6, 0), (6, 4), (6, 5), (7, 0), (7, 1),
    #     (7, 2), (7, 3), (8, 0), (8, 2), (9, 2), (10, 0), (10, 4),
    #     (10, 5), (11, 0), (12, 0), (12, 3), (13, 0), (13, 1), (13, 2),
    #     (13, 3), (16, 5), (16, 6), (17, 0), (17, 1), (19, 0), (19, 1),
    #     (21, 0), (21, 1), (25, 23), (25, 24), (27, 2), (27, 23),
    #     (27, 24), (28, 2), (29, 23), (29, 26), (30, 1), (30, 8),
    #     (31, 0), (31, 24), (31, 25), (31, 28), (32, 2), (32, 8),
    #     (32, 14), (32, 15), (32, 18), (32, 20), (32, 22), (32, 23),
    #     (32, 29), (32, 30), (32, 31), (33, 8), (33, 9), (33, 13),
    #     (33, 14), (33, 15), (33, 18), (33, 19), (33, 20), (33, 22),
    #     (33, 23), (33, 26), (33, 27), (33, 28), (33, 29), (33, 30),
    #     (33, 31), (33, 32)]
    # add edges two lists of nodes: src and dst
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    # edges are directional in DGL; make them bi-directional
    # g.add_edges(dst, src)
    g.edata['e_label'] = torch.randint(low=0, high=1, size=(1,6)).squeeze(0)
    g.apply_edges(lambda edges: {'node_id_pair': torch.cat((edges.src['n_id'], edges.dst['n_id']), dim=-1)})
    print(g.edge_ids(2,1))
    multi_edge_node_pairs(graph=g)

    return g

def fast_sampler(num_samples, sample_size, elements, probabilities):
    random_shift = np.random.random(len(probabilities))
    random_shift /= random_shift.sum()
    shifted_probabilties = random_shift - probabilities
    return elements[np.argpartition(shifted_probabilties, sample_size)[:sample_size][0]]
    # replicate probabilities as many times as `num_samples`
    # replicated_probabilities = np.tile(probabilities, (num_samples, 1))
    # random_shifts = np.random.random(replicated_probabilities.shape)
    # random_shifts /= random_shifts.sum(axis=1)[:, np.newaxis]
    # shifted_probabilities = random_shifts - replicated_probabilities
    # return np.argpartition(shifted_probabilities, sample_size, axis=1)[:, :sample_size]

#=============================================================
class WeightedRandomGenerator(object):
    def __init__(self, weights):
        self.totals = []
        running_total = 0
        for w in weights:
            running_total += w
            self.totals.append(running_total)

    def next(self):
        rnd = random.random() * self.totals[-1]
        return bisect.bisect_right(self.totals, rnd)

    def __call__(self):
        return self.next()
#=============================================================

def fast_sampler_new(sample_size, elements, probabilities):
    x = np.random.rand()
    cum = 0
    for i, p in enumerate(probabilities):
        cum += p
        if x < cum:
            break
    return elements[i]

def data_frame_test():
    data = np.random.randint(0, 10, (1000, 5))
    df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D', 'E'])
    groups = df.groupby(['A', 'B'])
    for i, g in groups:
        print(i, g.shape[0])

    print(df.head(5))

# def build_karate_club_graph():
#     g = dgl.DGLGraph(multigraph=True)
#     node_id = torch.arange(0, 4, dtype=torch.long).view(-1, 1)
#     # add 34 nodes into the graph; nodes are labeled from 0~33
#     g.add_nodes(4)
#     g.ndata['n_id'] = node_id
#     # all 78 edges as a list of tuples
#     edge_list = [(1,0), (2, 0), (2, 1), (3, 0), (2,1), (2,1)]
#     # add edges two lists of nodes: src and dst
#     src, dst = tuple(zip(*edge_list))
#     g.add_edges(src, dst)
#     # edges are directional in DGL; make them bi-directional
#     # g.add_edges(dst, src)
#     g.edata['e_label'] = torch.randint(low=0, high=1, size=(1,6)).squeeze(0)
#     g.apply_edges(lambda edges: {'node_id_pair': torch.cat((edges.src['n_id'], edges.dst['n_id']), dim=-1)})
#     return g

def tree_test():
    g = build_karate_club_graph()
    tree_df = tree_extractor(g, hop_num=2)
    tree_df = tree_to_shortest_path(tree_df=tree_df, graph=g, hop_num=2)
    entity_path_to_relation(data=tree_df,graph=g, hop_num=2)

def _sparse_normalize_rows(mat):
    """
    Normalize a sparse CSR matrix row-wise (each row sums to 1)
    If a row is all 0's, it remains all 0's

    +++++++++++++++++++++
    adjacent matrix A ==> random walk normalized laplacian matrix D^(-1)A
    +++++++++++++++++++++
    Parameters
    ----------
    mat : scipy.sparse.csr matrix
        Matrix in CSR sparse format
    Returns
    -------
    out : scipy.sparse.csr matrix
        Normalized matrix in CSR sparse format
    """
    # print(mat) format of mat ===> (row_idx, col_idx), weight
    n_nodes = mat.shape[0]
    # Normalize Adjacency matrix to transition matrix
    # Diagonal of the degree matrix is the sum of nonzero elements
    degrees_div = np.array(np.sum(mat, axis=1)).flatten() # out-degrees
    # This is equivalent to inverting the diag mat
    # weights are 1 / degree
    degrees = np.divide(
        1,
        degrees_div,
        out=np.zeros_like(degrees_div, dtype=float),
        where=(degrees_div != 0)
    )
    # construct sparse diag mat
    # to broadcast weights to adj mat by dot product
    D = sparse.dia_matrix((n_nodes, n_nodes), dtype=np.float64)
    D.setdiag(degrees)
    # premultiplying by diag mat is row-wise mul
    return sparse.csr_matrix(D.dot(mat))

# data_frame_test()

tree_test()
# x = np.array([1,2,3,4,5,1,2,3,4])
# idx = np.unique(x, return_inverse=True, return_index=True)
# print(idx)

# graph = build_karate_club_graph()
#
# walkLens = np.array([1,1,1,2])
# lengths = torch.from_numpy(walkLens) + 1

# edge_list = [(1,0), (2, 0), (2, 1), (3, 0), (2, 0)]
#
# from_list = [x[0] for x in edge_list]
# to_list = [x[1] for x in edge_list]
#
# # print(from_list)
#
# edge_ids = graph.edge_ids(from_list, to_list, force_multi=True)

# nxGraph = graph.to_networkx()
# # A = nx.adj_matrix(nxGraph)
#
# # print(A)
#
# print('fff')
# nx_dgl_graph = dgl.DGLGraph(nxGraph)
# #
# y = nx_dgl_graph.adjacency_matrix_scipy(transpose=True, return_edge_ids=False)
#
# print(y)

# print('fff')
# x = graph.adjacency_matrix_scipy(transpose=True, return_edge_ids=False)
# y = _sparse_normalize_rows(x)
#
# nxGraph = graph.to_networkx()
# A = nx.adj_matrix(nxGraph)
# z = _sparse_normalize_rows(A)
# print(y)
# print('ffff')
# print(z)
# print(x)

# print(edge_ids)
#
# print(len(edge_ids), len(to_list))

# from time import time
# elements = list(range(10))
# probability = np.array([1.0]*10)*1.0#np.random.random(10)
# # probability[2] = 0
#
# sampler = WeightedRandomGenerator(weights=probability)
#
# probability /= probability.sum()
# start = time()
# # for i in range(10):
# for i in range(10000):
#     x = (fast_sampler(1, 1, elements, probability))
# # y = x.reshape((1, 10000))
# # print(np.unique(y[0], return_counts=True))
# print(time() - start)
#
# start = time()
# for i in range(10000):
#     z = np.random.choice(elements, p=probability)
# #print(np.unique(z, return_counts=True))
# print(time() - start)
#
# start = time()
# # for i in range(10):
# samp_array = []
# for i in range(10000):
#     x = sampler.next()
#     samp_array.append(x)
# # y = x.reshape((1, 10000))
# # print(np.unique(y[0], return_counts=True))
# samp_array = np.array(samp_array)
# print(np.unique(samp_array, return_counts=True))
# print(time() - start)
#
# # start = time()
# # for i in range(10000):
# #     (fast_sampler_new(1, elements, probability))
# # print(time() - start)
#
#
#
#
#
#
# # Test Graph
# # G = nx.generators.classic.wheel_graph(100)
# # G = build_karate_club_graph()
# # # adj_matrix = G.adjacency_matrix_scipy(transpose=True,return_edge_ids=False)
# # # print(adj_matrix)
# # # print(G.ndata['n_id'].squeeze(-1).tolist())
# # # G = G.to_networkx()
# #
# # # print(G.edge_ids([1,2], [0,0]))
# #
# # kg2pattern = KG2RelPattern()
# # walks = kg2pattern.extractor(G)
# # x = walks.drop_duplicates()
# # print(x['walk_len'].to_numpy(), type(x['walk_len']))
# #
# # print(walks)
# # print(x)
# # G.edge_id(1,0)
#
#
# # Fit embedding model to graph
# # g2v = Node2Vec()
#
#
# # g2v.fit(G)  # way faster than other node2vec implementations
# #
# # # query embeddings for node 42
# # print(g2v.predict(0))
# #
# # # Save model to gensim.KeyedVector format
# # g2v.save("wheel_model.bin")
# #
# # # load in gensim
# # from gensim.models import KeyedVectors
# #
# # model = KeyedVectors.load_word2vec_format("wheel_model.bin")
# # model[str(1)]  # need to make nodeID a str for gensim