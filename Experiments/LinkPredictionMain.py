import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


import argparse
import json
import logging
import os
import random

import numpy as np
from KGEmbedUtils.kgdataUtils import dataloader
from KGEmbedUtils.utils import set_seeds
from KGEmbedUtils.ioutils import load_json_as_data_frame
from CoRelKGModel.triple2pathDataLoader import construct_train_iterator
from KGBERTPretrain.bert import BERT
from ModelTrainer.LinkPredictionTrainer import LinkPredictionTrainer
from KGEmbedUtils.ioutils import load_model

def train():

    parser = argparse.ArgumentParser(
        description='Testing Knowledge Graph Embedding Models',
    )
    parser.add_argument('--input_data_path', type=str, default='../SeqData/Entity2PathWN18RR')
    parser.add_argument("-o", "--output_path", required=False, type=str, default='../SavedModel/WN18RR/LinkPred/',
                        help="model save path")
    parser.add_argument('--ent_data_name', type=str, default='kg_entity_to_tree_path_hop_3.json')
    parser.add_argument('--kg_data_name', type=str, default='WN18RR')
    parser.add_argument('--bert_model_name', type=str, default='pairnum_5000000__hid_256_layer_6_heads_4_bs_64_lr_1e-05_l2_1e-05_class_num_2_bert_acc_84.18.ep4')
    parser.add_argument('-n', '--negative_sample_size', default=32, type=int)
    parser.add_argument('-d', '--hidden_dim', default=256, type=int)
    parser.add_argument('-g', '--gamma', default=0.1, type=float)
    parser.add_argument('-b', '--batch_size', default=1, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)

    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
    parser.add_argument("--adam_weight_decay", type=float, default=0.001, help="weight_decay of adam")
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-gpu', '--num_gpu', default=0, type=int)
    parser.add_argument('--max_steps', default=500000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)

    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument("--rand_seed", type=int, default=2019, help="random seed")
    parser.add_argument("--pool_type", type=str, default='min', help="pool type")

    args = parser.parse_args()
    for key, value in vars(args).items():
        print('Parameter setting {} = {}'.format(key, value))
    seed = args.rand_seed
    set_seeds(seed)
    print("Loading Train Dataset", args.kg_data_name)
    data = dataloader(args.kg_data_name)
    num_nodes = data.num_entities
    num_relations = data.num_relation
    train_data = data.train
    relations = data.rel_dict_rev
    print("Number of entities = {}\nNumber of Edges = {}\nNumber of relations = {}.".format(num_nodes,
                                                                                            train_data.shape[0], num_relations))
    valid_data = data.valid
    test_data = data.test
    print('Train: {}, valid: {}, test: {}'.format(train_data.shape[0], valid_data.shape[0], test_data.shape[0]))
    all_data = np.concatenate([train_data, valid_data, test_data], axis=0)
    all_true_triples = [tuple(x) for x in all_data.tolist()]
    test_triples = [tuple(x) for x in test_data.tolist()]
    ent2path_data = load_json_as_data_frame(os.path.join(args.input_data_path, args.ent_data_name))

    print("Creating Dataloader")
    train_triples = [tuple(x) for x in train_data.tolist()]
    train_iterator = construct_train_iterator(train_triples=train_triples, num_entity=num_nodes,
                                              num_relations=num_relations, ent2path=ent2path_data,
                                              negative_sample_size=args.negative_sample_size, batch_size=args.batch_size)

    print("Building BERT model")
    BERT_Path = '../SavedModel'
    bert_model = BERT(head_num=4, n_layers=6, hidden=256, rel_vocab_size=11, max_rel_len=7)
    bert_model = load_model(bert_model, os.path.join(BERT_Path, args.kg_data_name, args.bert_model_name))
    for param in bert_model.parameters(recurse=True):
        param.requires_grad = False

    print('Loading bert model = {}'.format(os.path.join(BERT_Path, args.kg_data_name, args.bert_model_name)))
    trainer = LinkPredictionTrainer(bert=bert_model,num_relations=num_relations, num_entity=num_nodes, num_gpu=args.num_gpu,
                                    lr=args.learning_rate, weight_decay=args.adam_weight_decay, pool_type=args.pool_type,
                                    gamma=args.gamma, hid_dim=args.hidden_dim, train_dataloader=train_iterator)

    save_path = args.output_path + 'link_prediction_hid_' + str(args.hidden_dim) + '_reg_' + str(args.regularization) + \
                '_neg_size_' + str(args.negative_sample_size) + '_bs_' + str(args.batch_size) + '_lr_' + str(args.learning_rate) + \
                '_gamma_' + str(args.gamma) + '_pool_type_' + args.pool_type + '_lp'
    trainer.train(args.max_steps, save_path=save_path)

if __name__ == '__main__':
    train()