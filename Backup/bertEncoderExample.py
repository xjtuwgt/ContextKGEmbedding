import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import torch
from KGEmbedUtils.utils import set_seeds

import numpy as np
import torch
from KGEmbedUtils.kgdataUtils import dataloader
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pandas import DataFrame
from time import time
from CoRelKGModel.triple2pathDataLoader import construct_train_iterator
import pandas as pd



if __name__ == '__main__':
    import networkx as nx
    from KGEmbedUtils.ioutils import load_json_as_data_frame

    entity_save_path = '../SeqData/Entity2PathWN18RR/'
    # entity_save_path = '../SeqData/Entity2PathFB_15K_237/'
    # data = dataloader('FB15k-237')
    data = dataloader('WN18RR')
    num_nodes = data.num_entities
    num_relations = data.num_relation
    train_data = data.train
    relations = data.rel_dict_rev
    set_seeds(2019)
    print("Number of entities = {}\nNumber of Edges = {}\nNumber of relations = {}.".format(num_nodes,
                                                                                            train_data.shape[0], num_relations))
    valid_data = data.valid
    test_data = data.test
    print('Train: {}, valid: {}, test: {}'.format(train_data.shape[0], valid_data.shape[0], test_data.shape[0]))
    all_data = np.concatenate([train_data, valid_data, test_data], axis=0)
    all_true_triples = [tuple(x) for x in all_data.tolist()]
    test_triples = [tuple(x) for x in test_data.tolist()]
#
#     # start = time()
    ent2path = load_json_as_data_frame(entity_save_path + 'kg_entity_to_tree_path_hop_3.json')
    # test_datat_list = construct_test_iterator(test_triples=test_triples, all_true_triples=all_true_triples, num_entity=num_nodes, epoch_idx=0, batch_size=2,
    #                                           num_relations=num_relations, ent2path=ent2path)

    # start = time()
    # idx = 0
    # for batch in test_datat_list[0]:
    #     print(batch['path'].shape)
    #     print(idx, time() - start)
    #     idx = idx + 1

    BERT_MODELname = '../SavedModel/WN18RR/100000_hid_256_layer_3_heads_4_bs_32_lr_1e-05_l2_0.001_bert.ep5'

    from KGBERTPretrain.bert import BERT
    from KGEmbedUtils.ioutils import load_model

    # bert_model = BERT(rel_vocab_size=11, max_rel_len=17, hidden=256, n_layers=3, head_num=4, drop_out=0.1)
    #
    # bert_model = load_model(bert_model, BERT_MODELname)
    bert_model: BERT = torch.load(BERT_MODELname)

    # bert_model()

    train_triples = [tuple(x) for x in train_data.tolist()]
    train_iterator = construct_train_iterator(train_triples=train_triples, num_entity=num_nodes, num_relations=num_relations, ent2path=ent2path,
                                              negative_sample_size=5, batch_size=1)
    # #=================================================================
    #
    #
    # # print(data[''])
    # # #=================================================================
    start = time()
    for i in range(2):
        batch_i = next(train_iterator)
        paths, path_mask, path_pos = batch_i['path'], batch_i['path_mask'], batch_i['path_position']
        # print(batch_i['path'].shape)
        emb = bert_model.forward(rel_seq=paths, seq_mask=path_mask, pos_seq=path_pos)
        print(emb[:,0].shape)
        # print(emb.shape)
        # # print(pos.shape, neg.shape)
        # print(i, time() - start)

