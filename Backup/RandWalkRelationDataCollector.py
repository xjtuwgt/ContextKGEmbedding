import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Backup.KGUtils import build_graph_from_triples_directed
from dgl import DGLGraph
from KGEmbedUtils.kgdataUtils import dataloader
from KGEmbedUtils.utils import set_seeds
from KGEmbedUtils.ioutils import *

def random_walk_relation_path(graph: DGLGraph, epochs, walk_len, save_seq_data_path, seed=2019):
    from RandomWalk.RandWalkOnGraph import KG2RelPattern
    g2v = KG2RelPattern(epochs=epochs, walklen=walk_len, walklen_var_flag=True)
    walks, relation_walks, node_pair_walks = g2v.extractor(graph)
    print(walks.shape, relation_walks.shape)
    # # ++++++++++++++++++++++++++++++++++++
    walk_data_file_name = 'Walks_Epoch_'+str(epochs) + "_walklen_"+str(walk_len) + '_seed_' + str(seed)
    rel_pair_data_file_name = 'Relation_Walk_Epoch_' + str(epochs) + "_walklen_" + str(walk_len) + '_seed_' + str(seed)
    node_pair_data_file_name = 'NodePair_Walk_Epoch_' + str(epochs) + "_walklen_" + str(walk_len) + '_seed_' + str(seed)
    save_random_walk_data(walks, relation_walks, node_pair_walks, walk_file_name= save_seq_data_path + walk_data_file_name,
                          rel_walk_file_name= save_seq_data_path + rel_pair_data_file_name, node_pair_file_name=save_seq_data_path +node_pair_data_file_name)

def save_random_walk_data(walks: DataFrame, relation_walks: DataFrame, node_pair_to_path:DataFrame, walk_file_name, rel_walk_file_name, node_pair_file_name):
    save_to_HDF(walks, walk_file_name + '.h5')
    save_to_json(walks, walk_file_name + '.json')
    save_to_json(relation_walks, rel_walk_file_name + '.json')
    save_to_json(node_pair_to_path, node_pair_file_name +'.json')
    print(rel_walk_file_name + '.json')
    print(node_pair_file_name + '.json')

if __name__ == '__main__':
    # data = dataloader('FB15k-237')
    data = dataloader('WN18RR')
    num_nodes = data.num_entities
    num_relations = data.num_relation
    train_data = data.train
    relations = data.rel_dict_rev
    print("Number of entities = {}\nNumber of Edges = {}\nNumber of relations = {}.".format(num_nodes, train_data.shape[0], num_relations))
    graph, rel, in_degree, out_degree, multi_edge_nodes = build_graph_from_triples_directed(num_nodes=num_nodes,
                                           num_relations=num_relations,
                                           triples=train_data.transpose())

    # print((out_degree > 0).sum())

    ## Step 1: random walk
    seed = 2019
    set_seeds(seed)
    epochs = 1000
    walk_len = 8
    seq_data_path = '../SeqData/RandWalk_WN18RR/'
    # seq_data_path = '../SeqData/RandWalk_FB15k_237/'
    random_walk_relation_path(graph=graph, epochs=epochs, walk_len=walk_len, save_seq_data_path=seq_data_path, seed=seed)