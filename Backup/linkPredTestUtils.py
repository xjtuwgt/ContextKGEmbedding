import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


import argparse
import torch
import numpy as np
from KGEmbedUtils.kgdataUtils import dataloader
from KGEmbedUtils.ioutils import load_json_as_data_frame, save_to_json
from KGEmbedUtils.ioutils import load_model
from time import time

from CoRelKGModel.triple2pathDataLoader import construct_test_iterator
from CoRelKGModel.kgePrediction import ContextKGEModel
from KGBERTPretrain.bert import BERT
import pandas as pd

def lp_eval_step(model: ContextKGEModel, test_dataset_list, device, epoch_idx):
    head_predictions = []
    tail_predictions = []
    start = time()
    batch_idx = 0
    with torch.no_grad():
        for test_dataset in test_dataset_list:
            for batch in test_dataset:
                mode = batch['mode']
                batch = {key: value.to(device) for key, value in batch.items() if key != 'mode'}
                paths, path_mask, path_position, path_batch_num, path_total_num, bias = batch['path'], batch['path_mask'], \
                                                                                          batch['path_position'], \
                                                                                          batch['path_bias_idx'], \
                                                                                          batch['path_num'], batch['batch_bias']
                batch_i = (paths, path_position, path_mask, path_batch_num)
                p_scores, n_scores = model.model_test(batch_i)
                p_scores, n_scores = torch.sigmoid(p_scores), torch.sigmoid(n_scores) + bias
                score_tuple = (p_scores.cpu().numpy(), n_scores.cpu().numpy())
                # for x in score_tuple:
                #     print(x.shape)
                if mode == 'head_batch':
                    head_predictions.append(score_tuple)
                elif mode=='tail_batch':
                    tail_predictions.append(score_tuple)
                else:
                    raise ValueError('Negative batch mode %s not supported' % mode)
                batch_idx = batch_idx + 1
                if batch_idx % 100 == 0:
                    print('Epoch {}: batch {} in {} seconds'.format(epoch_idx, batch_idx, time()-start))
                    start = time()

    return pd.DataFrame(head_predictions, columns=['p', 'n']), pd.DataFrame(tail_predictions, columns=['p', 'n'])


def epoch_test_data_loader(epoch_idx, shared_data:dict):
    test_dataset_list = construct_test_iterator(test_triples=shared_data['test_triple'], all_true_triples=shared_data['all_true_triples'],
                                                number_workers=shared_data['num_workers'], ent2path=shared_data['ent2path'],
                                                chunk_size=shared_data['chunk_size'], num_entity=shared_data['num_entity'],
                                                num_relations=shared_data['num_relation'], batch_size=1, epoch_idx=epoch_idx)
    return test_dataset_list

def link_prediction_evaluation(model: ContextKGEModel, test_triples, all_true_triples, num_entity,
              num_relation, ent2_path_data, chunk_size,
              num_workers, device, save_path):
    '''
    Evaluate the model on test or valid datasets
    '''
    model.eval()
    epoch_num = num_entity // chunk_size if num_entity % chunk_size == 0 else (num_entity // chunk_size + 1)
    shared_data = {'test_triple': test_triples, 'all_true_triples': all_true_triples, 'num_entity': num_entity,
                   'num_relation': num_relation, 'chunk_size': chunk_size, 'ent2path': ent2_path_data, 'num_workers': num_workers}

    for epoch_idx in range(epoch_num):
        test_data_list = epoch_test_data_loader(epoch_idx=epoch_idx, shared_data=shared_data)
        head_df, tail_df = lp_eval_step(model=model, test_dataset_list=test_data_list, device=device, epoch_idx=epoch_idx)
        save_to_json(head_df, save_path + '_test_size_' + str(len(test_triples)) + '_head_'+ str(epoch_idx) +'.json')
        save_to_json(tail_df, save_path + '_test_size_' + str(len(test_triples)) + '_tail_' + str(epoch_idx) + '.json')
        print('Epoch {} has been completed evaluated'.format(epoch_idx))

                    # argsort = torch.argsort(score, dim=1, descending=True)
                    # if mode == 'head-batch':
                    #     positive_arg = positive_sample[:, 0]
                    # elif mode == 'tail-batch':
                    #     positive_arg = positive_sample[:, 2]
                    # else:
                    #     raise ValueError('mode %s not supported' % mode)
                    #
                    # for i in range(batch_size):
                    #     # Notice that argsort is not ranking
                    #     ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                    #     assert ranking.size(0) == 1
                    #
                    #     # ranking + 1 is the true ranking used in evaluation metrics
                    #     ranking = 1 + ranking.item()
                    #     logs.append({
                    #         'MRR': 1.0 / ranking,
                    #         'MR': float(ranking),
                    #         'HITS@1': 1.0 if ranking <= 1 else 0.0,
                    #         'HITS@3': 1.0 if ranking <= 3 else 0.0,
                    #         'HITS@10': 1.0 if ranking <= 10 else 0.0,
                    #     })
                    # if step % args.test_log_steps == 0:
                    #     logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

    #                 step += 1
    #         metrics = {}
    #     for metric in logs[0].keys():
    #         metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
    # return metrics

def validation_start():
    parser = argparse.ArgumentParser(description='Testing Knowledge Graph Embedding Models')

    parser.add_argument('--input_ent2path_path', type=str, default='../SeqData/Entity2PathWN18RR')
    parser.add_argument("--input_model_path", required=False, type=str, default='../SavedModel/WN18RR/LinkPred/',
                        help="model path")
    # parser.add_argument("--bert_path", required=False, type=str, default='../SavedModel/WN18RR/', help="bert model path")
    # parser.add_argument('--bert_model_name', type=str,
    #                     default='10000000_hid_256_layer_6_heads_4_bs_64_lr_1e-05_l2_1e-05_class_num_2_bert_acc_84.07.ep13')
    parser.add_argument('--ent_data_name', type=str, default='kg_entity_to_tree_path_hop_3.json')
    parser.add_argument('--kg_data_name', type=str, default='WN18RR')
    parser.add_argument('--lp_model_name', type=str, default='link_prediction_hid_256_reg_0.0_neg_size_32_bs_4_lr_0.0001_gamma_1_lp_loss_75.894.bt105000')
    parser.add_argument('-d', '--hidden_dim', default=256, type=int)
    parser.add_argument('-b', '--test_batch_size', default=1, type=int)
    parser.add_argument('-c', '--chunk_size', default=200, type=int)
    parser.add_argument('--num_workers', default=1, type=float)
    parser.add_argument('-g', '--gamma', default=1, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=8, type=int)
    parser.add_argument('-gpu', '--num_gpu', default=0, type=int)
    parser.add_argument('--output_path', default='../SavedResults/WN18RR/', type=str)

    args = parser.parse_args()
    for key, value in vars(args).items():
        print('Parameter setting {} = {}'.format(key, value))

    print("Step 1: Loading Validation Dataset", args.kg_data_name)
    data = dataloader(args.kg_data_name)
    num_nodes = data.num_entities
    num_relations = data.num_relation
    train_data = data.train
    relations = data.rel_dict_rev
    print("Number of entities = {}\nNumber of Edges = {}\nNumber of relations = {}.".format(num_nodes,
                                                                                            train_data.shape[0],
                                                                                            num_relations))
    valid_data = data.valid
    test_data = data.test
    print('Train: {}, valid: {}, test: {}'.format(train_data.shape[0], valid_data.shape[0], test_data.shape[0]))
    all_data = np.concatenate([train_data, valid_data, test_data], axis=0)
    all_true_triples = [tuple(x) for x in all_data.tolist()]
    test_triples = [tuple(x) for x in test_data.tolist()]
    valid_triples = [tuple(x) for x in valid_data.tolist()]
    ent2path_data = load_json_as_data_frame(os.path.join(args.input_ent2path_path, args.ent_data_name))

    print("Step 2: Loading Model for", args.kg_data_name)
    bert_model = BERT(head_num=4, n_layers=6, hidden=256, rel_vocab_size=11, max_rel_len=7)
    prediction_model_name = args.input_model_path + args.lp_model_name
    pred_model = ContextKGEModel(bert_path_encoder=bert_model, num_entity=num_nodes,
                                 num_relation=num_relations, gamma=args.gamma, hidden_dim=bert_model.hidden)
    pred_model = load_model(pred_model, prediction_model_name)
    print("Step 3: Link Prediction evaluation on {}", args.kg_data_name)
    device_ids = None
    if args.num_gpu >= 1:
        print('Using GPU!')
        if args.num_gpu > 1:
            device_ids = range(args.num_gpu)
        aa = []
        for i in range(args.num_gpu - 1, -1, -1):
            device = torch.device('cuda: % d' % i)
            aa.append(torch.rand(1).to(device))  # a place holder
    else:
        device = 'cpu'
        print('Using CPU!')
    for param in pred_model.parameters(recurse=True):
        param.requires_grad = False
    pred_model = pred_model.to(device)
    link_prediction_evaluation(model=pred_model, test_triples=test_triples[:10], all_true_triples=all_true_triples,
                               num_relation=num_relations, num_entity=num_nodes, chunk_size=args.chunk_size,
                               num_workers=args.num_workers, ent2_path_data=ent2path_data, device=device, save_path=args.output_path)


if __name__ == '__main__':
    validation_start()
