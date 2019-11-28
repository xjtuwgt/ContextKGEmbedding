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
import torch
from GraphUtils.KGUtils import multi_edge_node_pairs

### TODO: could drop gensim dependency by making coocurence matrix on the fly
###       instead of random walks and use GLoVe on it.
###       but then why not just use GLoVe on Transition matrix?
# Gensim triggers automatic useless warnings for windows users...
warnings.simplefilter("ignore", category=UserWarning)
import gensim

warnings.resetwarnings()


# TODO: Organize Graph method here
# Layout nodes by their 1d embedding's position


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def _csr_random_walk(Tdata, Tindptr, Tindices,
                     sampling_nodes,
                     walklen, walklen_var_flag=False):
    """
    Create random walks from the transition matrix of a graph
        in CSR sparse format
    NOTE: scales linearly with threads but hyperthreads don't seem to
            accelerate this linearly

    Parameters
    ----------
    Tdata : 1d np.array
        CSR data vector from a sparse matrix. Can be accessed by M.data
    Tindptr : 1d np.array
        CSR index pointer vector from a sparse matrix.
        Can be accessed by M.indptr
    Tindices : 1d np.array
        CSR column vector from a sparse matrix.
        Can be accessed by M.indices
    sampling_nodes : 1d np.array of int
        List of node IDs to start random walks from.
        Is generally equal to np.arrange(n_nodes) repeated for each epoch
    walklen : int
        length of the random walks
    Returns
    -------
    out : 2d np.array (n_walks, walklen)
        A matrix where each row is a random walk,
        and each entry is the ID of the node
    """
    n_walks = len(sampling_nodes)
    # res = np.empty((n_walks, walklen), dtype=np.int64)
    # res = np.full((n_walks, walklen), fill_value=-1, dtype=np.int64)
    res = np.full((n_walks, walklen + 1), fill_value=-1, dtype=np.int64)
    for i in numba.prange(n_walks):
        # Current node (each element is one walk's state)
        state = sampling_nodes[i]
        # ++++++++++++++++++++++++++++++++++++++++++++++
        """
        Add the parameters to control the walklen
        """
        # ++++++++++++++++++++++++++++++++++++++++++++++
        if walklen_var_flag:
            var_walk_len = np.random.randint(low=2, high=walklen+1)
        else:
            var_walk_len = walklen
        # ++++++++++++++++++++++++++++++++++++++++++++++
        # print(var_walk_len)
        # +++++++
        true_walk_len = 0
        # +++++++
        for k in range(var_walk_len - 1): # Generate random walks with different length
        # for k in range(walklen - 1):
            # Write state
            res[i, k] = state
            # Find row in csr indptr
            start = Tindptr[state]
            end = Tindptr[state + 1]
            # transition probabilities
            p = Tdata[start:end]
            # cumulative distribution of transition probabilities
            cdf = np.cumsum(p)
            # +++++++++++Original+++++++++++++++++++
            # print(cdf.size == 0)
            # # Random draw in [0, 1] for each row
            # # Choice is where random draw falls in cumulative distribution
            # draw = np.random.rand()
            # # Find where draw is in cdf
            # # Then use its index to update state
            # next_idx = np.searchsorted(cdf, draw)
            # # Winner points to the column index of the next node
            # state = Tindices[start + next_idx]
            # +++++++++++Original+++++++++++++++++++
            if cdf.size > 0:
                # Random draw in [0, 1] for each row
                # Choice is where random draw falls in cumulative distribution
                draw = np.random.rand()
                # Find where draw is in cdf
                # Then use its index to update state
                next_idx = np.searchsorted(cdf, draw)
                # Winner points to the column index of the next node
                state = Tindices[start + next_idx]
                #++++++++
                true_walk_len = true_walk_len + 1
                #++++++++
            else:
                state = -1 #There is no out-going edge from the current node
                break
        # Write final states
        # res[i, -1] = state
        if walklen_var_flag:
            res[i, var_walk_len - 1] = state
            #++++++++
        else:
            res[i, -2] = state
        #++++++++
        res[i, -1] = true_walk_len #last column count the true number of steps
        #++++++++
    return res


# TODO: This throws heap corruption errors when made parallel
#       doesn't seem to be illegal reads anywhere though...
@jit(nopython=True, nogil=True, fastmath=True)
def _csr_node2vec_walks(Tdata, Tindptr, Tindices,
                        sampling_nodes,
                        walklen,
                        return_weight,
                        neighbor_weight, walklen_var_flag=False):
    """
    Create biased random walks from the transition matrix of a graph
        in CSR sparse format. Bias method comes from Node2Vec paper.
    +++++++++++++
    2nd random walk
    +++++++++++++
    Parameters
    ----------
    Tdata : 1d np.array
        CSR data vector from a sparse matrix. Can be accessed by M.data
    Tindptr : 1d np.array
        CSR index pointer vector from a sparse matrix.
        Can be accessed by M.indptr
    Tindices : 1d np.array
        CSR column vector from a sparse matrix.
        Can be accessed by M.indices
    sampling_nodes : 1d np.array of int
        List of node IDs to start random walks from.
        Is generally equal to np.arrange(n_nodes) repeated for each epoch
    walklen : int
        length of the random walks
    return_weight : float in (0, inf]
        Weight on the probability of returning to node coming from
        Having this higher tends the walks to be
        more like a Breadth-First Search.
        Having this very high  (> 2) makes search very local.
        Equal to the inverse of p in the Node2Vec paper.
    neighbor_weight : float in (0, inf]
        Weight on the probability of visitng a neighbor node
        to the one we're coming from in the random walk
        Having this higher tends the walks to be
        more like a Depth-First Search.
        Having this very high makes search more outward.
        Having this very low makes search very local.
        Equal to the inverse of q in the Node2Vec paper.
    Returns
    -------
    out : 2d np.array (n_walks, walklen)
        A matrix where each row is a biased random walk,
        and each entry is the ID of the node
    """
    n_walks = len(sampling_nodes)
    # res = np.empty((n_walks, walklen), dtype=np.int64)
    res = np.full((n_walks, walklen), fill_value=-1, dtype=np.int64)
    for i in range(n_walks):
        # Current node (each element is one walk's state)
        # +++++++
        true_walk_len = 0
        # +++++++
        state = sampling_nodes[i]
        res[i, 0] = state
        # Do one normal step first
        # comments for these are in _csr_random_walk
        start = Tindptr[state]
        end = Tindptr[state + 1]
        p = Tdata[start:end]
        cdf = np.cumsum(p)
        if cdf.size > 0:
            draw = np.random.rand()
            next_idx = np.searchsorted(cdf, draw)
            state = Tindices[start + next_idx]
            # ++++++++++++++++++++++++++++++++++++++++++++++
            """
            Add the parameters to control the walklen
            """
            # ++++++++++++++++++++++++++++++++++++++++++++++
            if walklen_var_flag:
                var_walk_len = np.random.randint(low=2, high=walklen + 1)
            else:
                var_walk_len = walklen
            # ++++++++++++++++++++++++++++++++++++++++++++++
            for k in range(1, var_walk_len - 1):
            # for k in range(1, walklen - 1):
                # Write state
                res[i, k] = state
                # +++++++
                true_walk_len = true_walk_len + 1
                # +++++++
                # Find rows in csr indptr
                prev = res[i, k - 1]
                start = Tindptr[state]
                end = Tindptr[state + 1]
                start_prev = Tindptr[prev]
                end_prev = Tindptr[prev + 1]
                # Find overlaps and fix weights
                this_edges = Tindices[start:end]
                prev_edges = Tindices[start_prev:end_prev]
                p = np.copy(Tdata[start:end])
                ret_idx = np.where(this_edges == prev)
                p[ret_idx] = np.multiply(p[ret_idx], return_weight)
                for pe in prev_edges:
                    n_idx = np.where(this_edges == pe)[0]
                    p[n_idx] = np.multiply(p[n_idx], neighbor_weight)
                # Get next state
                cdf = np.cumsum(np.divide(p, np.sum(p)))
                if cdf.size > 0:
                    draw = np.random.rand()
                    next_idx = np.searchsorted(cdf, draw)
                    state = this_edges[next_idx]
                else:
                    state = -1
                    break
            # Write final states
            #res[i, k] = state
        #+++++++++++++++++++++++
        # res[i, -1] = state
        if walklen_var_flag:
            res[i, var_walk_len - 1] = state
        else:
            res[i, -2] = state
        #++++++++
        if state != -1:
            true_walk_len = true_walk_len + 1
        res[i, -1] = true_walk_len #last column count the true number of steps
        #+++++++++++++++++++++++
    return res


def make_walks(T,
               walklen=10,
               epochs=3,
               return_weight=1.,
               neighbor_weight=1.,
               threads=0, walklen_var_flag=False):
    """
    Create random walks from the transition matrix of a graph
        in CSR sparse format
    NOTE: scales linearly with threads but hyperthreads don't seem to
            accelerate this linearly
    Parameters
    ----------
    T : scipy.sparse.csr matrix
        Graph transition matrix in CSR sparse format
    walklen : int
        length of the random walks
    epochs : int
        number of times to start a walk from each nodes
    return_weight : float in (0, inf]
        Weight on the probability of returning to node coming from
        Having this higher tends the walks to be
        more like a Breadth-First Search.
        Having this very high  (> 2) makes search very local.
        Equal to the inverse of p in the Node2Vec paper.
    neighbor_weight : float in (0, inf]
        Weight on the probability of visitng a neighbor node
        to the one we're coming from in the random walk
        Having this higher tends the walks to be
        more like a Depth-First Search.
        Having this very high makes search more outward.
        Having this very low makes search very local.
        Equal to the inverse of q in the Node2Vec paper.
    threads : int
        number of threads to use.  0 is full use
    Returns
    -------
    out : 2d np.array (n_walks, walklen)
        A matrix where each row is a random walk,
        and each entry is the ID of the node
    """
    n_rows = T.shape[0]
    sampling_nodes = np.arange(n_rows)
    sampling_nodes = np.tile(sampling_nodes, epochs)
    if type(threads) is not int:
        raise ValueError("Threads argument must be an int!")
    if threads == 0:
        threads = numba.config.NUMBA_DEFAULT_NUM_THREADS
    threads = str(threads)
    try:
        prev_numba_value = os.environ['NUMBA_NUM_THREADS']
    except KeyError:
        prev_numba_value = threads
    # If we change the number of threads, recompile
    if threads != prev_numba_value:
        os.environ['NUMBA_NUM_THREADS'] = threads
        _csr_node2vec_walks.recompile()
        _csr_random_walk.recompile()
    if return_weight <= 0 or neighbor_weight <= 0:
        raise ValueError("Return and neighbor weights must be > 0")
    if (return_weight > 1. or return_weight < 1.
            or neighbor_weight < 1. or neighbor_weight > 1.):
        walks = _csr_node2vec_walks(T.data, T.indptr, T.indices,
                                    sampling_nodes=sampling_nodes,
                                    walklen=walklen,
                                    return_weight=return_weight,
                                    neighbor_weight=neighbor_weight,
                                    walklen_var_flag=walklen_var_flag)
    # much faster implementation for regular walks
    else:
        walks = _csr_random_walk(T.data, T.indptr, T.indices,
                                 sampling_nodes, walklen, walklen_var_flag)
    # set back to default
    os.environ['NUMBA_NUM_THREADS'] = prev_numba_value
    # print(walks)
    return walks


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
## ====================================================================================================================
## ====================================================================================================================

class KG2RelPattern():
    """
    Extract relation patterns from knowledge graph
    """

    def __init__(self, walklen: int = 10, epochs=10, return_weight=1.,
            neighbor_weight=1., threads=0, walklen_var_flag=False):
        """
        :param walk_len:
        :param epochs:
        :param return_weight:
        :param neighbor_weight:
        :param threads:
        """
        if type(threads) is not int:
            raise ValueError("Threads argument must be an int!")
        if walklen < 1 or epochs < 1:
            raise ValueError("Walklen and epochs arguments must be > 1")
        if return_weight < 0 or neighbor_weight < 0:
            raise ValueError("return_weight and neighbor_weight must be >= 0")
        self.walklen = walklen
        self.epochs = epochs
        self.return_weight = return_weight
        self.neighbor_weight = neighbor_weight
        if threads == 0:
            threads = numba.config.NUMBA_DEFAULT_NUM_THREADS
        self.threads = threads
        self.walklen_var_flag = walklen_var_flag


    def extractor(self, graph: DGLGraph, verbose=1):
        # node_names = list(nxGraph.nodes)
        # ++++++++++++++++++++
        node_names = graph.ndata['n_id'].squeeze(-1).tolist()
        # ++++++++++++++++++++
        if type(node_names[0]) not in [int, str, np.int32, np.int64]:
            raise ValueError("Graph node names must be int or str!")
        # Adjacency matrix
        # A = nx.adj_matrix(nxGraph)
        # ++++++++++++++++++++
        A = graph.adjacency_matrix_scipy(transpose=True, return_edge_ids=False)
        # ++++++++++++++++++++
        # print(A)
        n_nodes = A.shape[0]
        T = _sparse_normalize_rows(A)# random walk normalized laplacian
        walks_t = time.time()
        if verbose:
            print("Step 1: Making walks...", end=" ")
            # If node2vec graph weights not identity, apply them
        walks = make_walks(T, walklen=self.walklen, epochs=self.epochs,
                           return_weight=self.return_weight,
                           neighbor_weight=self.neighbor_weight,
                           threads=self.threads, walklen_var_flag=self.walklen_var_flag)
        if verbose:
            print(f"Done in time = {time.time() - walks_t:.2f} seconds")
            # print("Mapping Walk Names...", end=" ")

        # =================================================================================
        #                Random walk filter
        # 1) removing redundant
        # 2) removing one node random walk (that is, the out degree of the node is 0
        # 3) keeping all pairs of walks with same start and end, removing single walk
        # =================================================================================
        redundant_t = time.time()
        if verbose:
            print("Step 2: random walk filter...")
            print('\tThere are {} walks before removing redundants'.format(walks.shape[0]))
        indexes = np.arange(walks.shape[0])
        # pairs = (walks[:,0] + walks[indexes, walks[:, self.walklen]] * n_nodes).reshape((walks.shape[0], 1))
        start_nodes, end_nodes = walks[:,0].reshape((walks.shape[0], 1)), walks[indexes, walks[:, self.walklen]].reshape((walks.shape[0], 1))
        pairs = start_nodes + end_nodes * n_nodes
        walks = np.concatenate((walks, pairs, start_nodes, end_nodes), axis=1)
        ## step 1)
        walks = pd.DataFrame(data=walks) ## panda dataframe
        walks = walks.drop_duplicates()
        if verbose:
            print('\t{} walks achieved by dropping duplicates'.format(walks.shape[0]))
        ## step 2)
        walks = walks[walks[1] > -1] # Filter out the random walk with only one node.
        if verbose:
            print('\t{} walks achieved by filtering one node walks'.format(walks.shape[0]))
            print(f"\tRedundant filter is done in time = {time.time() - redundant_t:.2f} seconds")
        ## step 3)
        if verbose:
            print("Step 3: Relation pair constructed from walks...")
        rename_col_names = dict()
        for i in range(self.walklen):
            rename_col_names[i] = 'n_' + str(i)
        rename_col_names[self.walklen] = 'walk_len'
        rename_col_names[self.walklen + 1] = 'walk_pair'
        rename_col_names[self.walklen + 2] = 'start_node'
        rename_col_names[self.walklen + 3] = 'end_node'
        walks = walks.rename(columns=rename_col_names)
        # ============================================
        filter_single_time_t = time.time()
        walk_pair = walks['walk_pair'].value_counts()
        vals, counts = walk_pair.keys().tolist(), walk_pair.tolist()
        filter_pairs = [x for (x, y) in zip(vals, counts) if y > 1]
        walks['pair_or_not'] = walks['walk_pair'].isin(filter_pairs)
        if verbose:
            print("\t{} walks achieved by filtering single path in {:.2f} seconds"
                  .format(walks.shape[0], time.time() - filter_single_time_t))
        # ============================================
        map2relation_time_t = time.time()
        walks = self.node_walk_to_relation_walk(walks, graph)
        if verbose:
            print("\t{} walks are mapped to relation in {:.2f} seconds".format(walks.shape[0], time.time() - map2relation_time_t))
        # ============================================
        filter_pairs_num = [y for (x, y) in zip(vals, counts) if y > 1]
        mean_walk_num = np.array([x * (x - 1) / 2 for x in filter_pairs_num])
        if verbose:
            print('\t{} relation pairs are generated.'.format(mean_walk_num.sum()))
        # relation_pair_time = time.time()
        # relation_pair, num_pairs, num_groups = self.equivalent_relation_pair_all(walks)
        # if verbose:
        #     print("\t{} Relation pairs have been generated from {} "
        #           "groups in {:.2f} seconds".format(num_pairs, num_groups, time.time() - relation_pair_time))
        return walks, walks #walks, relation_pair

    def node_pair_with_multi_edges(self, walks: DataFrame, multi_edge_nodes: list):
        pair_in_names = []
        pair_names = []
        # start = time.time()
        for idx in range(self.walklen - 1):
            walks['pair_' + str(idx)] = list(zip(walks['n_'+str(idx)].values.tolist(), walks['n_'+str(idx + 1)].values.tolist()))
            walks['pair_in_' + str(idx)] = walks['pair_' + str(idx)].apply(lambda x: x in multi_edge_nodes)
            pair_in_names.append('pair_in_' + str(idx))
            pair_names.append('pair_' + str(idx))
        walks['multi_edge_flag'] = walks[pair_in_names].to_numpy().sum(axis=1) > 0
        walks = walks.drop(columns=pair_in_names + pair_names)
        # print('Data preparing time {}'.format(time.time() - start))
        return walks

    def node_walk_to_relation_walk(self, walks: DataFrame, graph: DGLGraph):
        multi_edge_nodes, _, _ = multi_edge_node_pairs(graph)
        for idx in range(self.walklen - 1):
            walks['r_' + str(idx)] = -1
        if len(multi_edge_nodes) > 0:
            walks = self.node_pair_with_multi_edges(walks, multi_edge_nodes)
            walks_groups = walks.groupby('multi_edge_flag')
            row_walks = walks_groups.get_group(True)
            row_walks = self.node_walk_to_relation_walk_by_row(row_walks, graph)
            col_walks = walks_groups.get_group(False)
            col_walks = self.node_walk_to_relation_walk_by_col(col_walks, graph)
            walks = pd.concat([row_walks, col_walks])
            walks = walks.drop(columns=['multi_edge_flag'])
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
                    group['r_'+str(i)] = edge_labels.long().tolist()
                groups.append(group)
        walks = pd.concat(groups)
        return walks

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




