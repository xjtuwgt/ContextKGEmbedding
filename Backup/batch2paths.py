import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import numpy as np
from time import time
from KGEmbedUtils.ioutils import load_json_as_data_frame
from KGEmbedUtils.kgdataUtils import dataloader
from Backup.TripleKGDataLoader import construct_test_iterator
from KGEmbedUtils.utils import set_seeds


def path_mask_and_position(paths):
    m, n = paths.shape[0], paths.shape[1]
    path_pad = paths.copy()
    path_pad[path_pad < 0] = 0
    path_mask = (paths >= 0).astype(np.int16)
    path_pos = np.argsort(-path_mask)
    first_idxs = np.repeat(np.array(np.arange(m)).reshape(m, 1), n, axis=1)
    path_position = np.repeat(np.array(np.arange(n)).reshape(1, n), m, axis=0)
    temp_idx = np.zeros(paths.shape, dtype=np.int16)
    temp_idx[first_idxs, path_pos] = path_position
    path_pos = path_position[first_idxs, temp_idx]
    path_pos[paths < 0] = 0
    return path_pad, path_mask, path_pos

if __name__ == '__main__':
    # # entity_save_path = '../SeqData/Entity2PathWN18RR/'
    # entity_save_path = '../SeqData/Entity2PathFB_15K_237/'
    # data = dataloader('FB15k-237')
    # # data = dataloader('WN18RR')
    # # data = dataloader('FB15K')
    # num_nodes = data.num_entities
    # num_relations = data.num_relation
    # train_data = data.train
    # relations = data.rel_dict_rev
    # # print(relations)
    # print("Number of entities = {}\nNumber of Edges = {}\nNumber of relations = {}.".format(num_nodes,
    #                                                                                         train_data.shape[0], num_relations))
    #
    # #=================================================================
    #
    # set_seeds(2019)
    # valid_data = data.valid
    # test_data = data.test
    # print('Train: {}, valid: {}, test: {}'.format(train_data.shape[0], valid_data.shape[0], test_data.shape[0]))
    # all_data = np.concatenate([train_data, valid_data, test_data], axis=0)
    # all_true_triples = [tuple(x) for x in all_data.tolist()]
    # test_triples = [tuple(x) for x in test_data.tolist()]
    #
    # data = load_json_as_data_frame(entity_save_path + 'kg_entity_to_tree_path_hop_1.json')
    # # start = time()
    # test_datat_list = construct_test_iterator(test_triples=test_triples, all_true_triples=all_true_triples, chunk_size=10, epoch_idx=0, batch_size=16,
    #                                           num_entity=num_nodes, num_relations=num_relations, ent2path=data)
    #
    #
    # # for idx, test in enumerate(test_datat_list):
    # #     for positive_sample, negative_sample, filter_bias, mode in test:
    # #         print(idx,filter_bias[0].sum())
    # # print('Constructing time = {}'.format(time() - start))
    # # train_triples = [tuple(x) for x in train_data.tolist()]
    # # train_iterator = construct_train_iterator(train_triples=train_triples, num_entity=num_nodes, num_relations=num_relations, negative_sample_size=5, batch_size=4, ent2path=data)
    #
    #
    # for col in data.columns:
    #     print(col)
    #
    # # print(data[''])
    # # #=================================================================
    # t_start = time()
    # # for i in range(2000):
    # #     start = time()
    # #     batch_i = next(train_iterator)
    # #     # print(batch_i[0])
    # #     # if i == 480:
    # #     # train_batch_to_paths(batch_i, data, num_relation=11)
    # #     # print(pos.shape, neg.shape)
    # #     print(i, time() - start)
    # # print(time()-t_start)
    #
    # batch_idx = 1
    # for test_data in test_datat_list:
    #     for batch in test_data:
    #         start = time()
    #         positive_sample, negative_sample, filter_bias, mode = batch
    #         # test_batch_to_paths(batch, data, num_relation=11)
    #         print('Batch {} time = {}'.format(batch_idx, time() - start))
    #         batch_idx = batch_idx + 1
    #     # batch_to_paths(batch, data, num_relation=11)
    #
    # print('total time = {}'.format(time() - t_start))

    paths = np.array([[9,-1, 3,0,5], [9,-1,-1,0,5], [9,1,3,-1,5]])
    print(paths)
    a, b, c = path_mask_and_position(paths)
    print(a)
    print(b)
    print(c)