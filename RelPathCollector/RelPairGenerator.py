import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from RandomWalk.RandWalkOnGraph import KG2RelPattern
from KGEmbedUtils.kgdataUtils import dataloader, build_graph_from_triples_bi_direction
from pandas import DataFrame
from KGEmbedUtils.utils import set_seeds
from KGEmbedUtils.ioutils import *
from time import time
import numpy as np


class RandomWalkGenerator(object):
    def __init__(self, kg_data_name: str, num_epochs, walk_len=4, random_seed=2019):
        set_seeds(random_seed)
        self.r_seed = random_seed
        self.kg_name = kg_data_name
        self.num_epochs = num_epochs
        self.walk_len = walk_len
        self.graph, self.rev_graph, self.num_entities = self.graph_construction(self.kg_name) # reverse graph to increase the diversity of paths
        self.rel_path_extractor = KG2RelPattern(walklen=walk_len, epochs=num_epochs, node_weight=True)
        return

    def graph_construction(self, kg_name):
        data = dataloader(kg_name)
        num_nodes = data.num_entities
        num_relations = data.num_relation
        train_data = data.train
        print("Number of entities = {}\nNumber of Edges "
              "= {}\nNumber of relations = {}.".format(num_nodes, train_data.shape[0], num_relations))
        graph, rev_graph = build_graph_from_triples_bi_direction(num_nodes=num_nodes, triples=train_data.transpose())
        return graph, rev_graph, num_nodes

    def rel_path_collection(self):
        print('A. Starting random walking on original graph....')
        start = time()
        walks = self.rel_path_extractor.extractor(self.graph)
        group_names = ['n_' + str(i) for i in range(walk_len)]
        group_names = group_names + ['start_node', 'end_node', 'walk_len']
        group_names = group_names + ['r_' + str(i) for i in range(walk_len - 1)]
        rand_walks= walks[group_names]
        print('Original random walk completed in {:.3f} seconds'.format(time()-start))
        start = time()
        print('B. Random walk to in/out pairs....')
        walk_pir, distinct_paths = self.rel_path_pair(walks=rand_walks, num_entities=self.num_entities, walk_len=self.walk_len)
        print('Entity to walk pairs in {:.3f} seconds'.format(time() - start))
        random_walk_name = 'RandWalk_walklen_' + str(self.walk_len) + '_epoch_' \
                           + str(self.num_epochs) + '_seed_' + str(self.r_seed)
        return rand_walks, walk_pir, distinct_paths, random_walk_name

    def rel_path_pair(self, walks: DataFrame, num_entities: int, walk_len):
        in_paths_list = [[] for _ in range(num_entities)]
        out_paths_list = [[] for _ in range(num_entities)]
        in_path_len_list = [[] for _ in range(num_entities)]
        out_path_len_list = [[] for _ in range(num_entities)]
        in_path_cnt_list = [[] for _ in range(num_entities)]
        out_path_cnt_list = [[] for _ in range(num_entities)]
        in_path_num, out_path_num = np.zeros(num_entities, dtype=np.int32), np.zeros(num_entities, dtype=np.int32)
        group_names = ['r_' + str(i) for i in range(walk_len - 1)]
        group_names.append('walk_len')
        relations = ['r_' + str(i) for i in range(walk_len - 1)]
        out_groups = walks.groupby('start_node')
        idx = 0
        for g_idx, group in out_groups:
            out_i_agg = group.groupby(group_names).size().reset_index(name='or_cnt')
            out_paths_list[g_idx] = out_i_agg[relations].to_numpy()
            out_path_len_list[g_idx] = out_i_agg['walk_len']
            out_path_cnt_list[g_idx] = out_i_agg['or_cnt']
            out_path_num[g_idx] = out_i_agg.shape[0]
            idx = idx + 1
            if idx % 5000 == 0:
                print('Idx {} for start node'.format(idx))
        idx = 0
        in_groups = walks.groupby('end_node')
        for g_idx, group in in_groups:
            in_i_agg = group.groupby(group_names).size().reset_index(name='ir_cnt')
            in_paths_list[g_idx] = in_i_agg[relations].to_numpy()
            in_path_len_list[g_idx] = in_i_agg['walk_len']
            in_path_num[g_idx] = in_i_agg.shape[0]
            in_path_cnt_list[g_idx] = in_i_agg['ir_cnt']
            idx = idx + 1
            if idx % 5000 == 0:
                print('Idx {} for end node'.format(idx))
        ent_list = [ent_id for ent_id in range(num_entities)]
        data = {'center': ent_list,  'i_num':in_path_num.tolist(), 'ir_path': in_paths_list, 'ir_len': in_path_len_list, 'ir_cnt': in_path_cnt_list,
                'o_num': out_path_num.tolist(), 'or_path': out_paths_list, 'or_len': out_path_len_list, 'or_cnt': out_path_cnt_list,}
        all_dist_paths = walks[relations].groupby(relations).size().reset_index(name='rel_cnt')
        data_frame = DataFrame(data=data)
        return data_frame, all_dist_paths

def save_random_walk(walks, pair_path_df, dist_path_df, walk_file_name):
    save_to_json(walks, walk_file_name + '_rand_walk.json')
    save_to_json(pair_path_df, walk_file_name + '_path_pair.json')
    save_to_json(dist_path_df, walk_file_name + '_uniq_path.json')
    print(walk_file_name)


if __name__ == '__main__':
    kg_name = 'WN18RR'
    path = '../SeqData/RelPathPair_WN18RR'
    walk_len = 5 #for link prediction/two-hop query, with tree depth = 3
    rand_seed = 2019
    epochs = 4000
    rand_walker = RandomWalkGenerator(kg_data_name=kg_name, walk_len=walk_len, num_epochs=epochs, random_seed=rand_seed)
    walks, pair_path, dist_paths, walk_name = rand_walker.rel_path_collection()
    save_random_walk(walks=walks, pair_path_df=pair_path, dist_path_df=dist_paths, walk_file_name=os.path.join(path, walk_name))

    # file_name = '../SeqData/Entity2PathWN18RR/kg_entity_to_tree_path_hop_3.json'
    # walks = load_json_as_data_frame(file_name)
    # for col in walks.columns:
    #     print(col)