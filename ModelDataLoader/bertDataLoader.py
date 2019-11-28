import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


import numpy as np
from random import random as rand
import torch
from KGEmbedUtils.kgdataUtils import dataloader
from KGEmbedUtils.ioutils import load_json_as_data_frame
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pandas import DataFrame
from time import time
from torch import Tensor
import pandas as pd
from KGEmbedUtils.utils import set_seeds

class BertTrainDataset(Dataset):
    def __init__(self, n_entity: int, n_relation, mode, sample_size: int, ent2path_data: DataFrame, all_true_paths: DataFrame=None):
        start = time()
        print('Data pre-processing...')
        self.n_entity = n_entity
        self.samp_size = sample_size
        self.n_relation = n_relation
        self.mode = mode
        self.len = n_entity
        self.ent2path_df, self.walk_len = self.get_ent2path_data_frame(ent2path_data)
        if all_true_paths is not None:
            self.true_path_dict = self.get_true_paths(all_true_paths)
            self.true_path_num = len(self.true_path_dict)
            self.true_paths = list(self.true_path_dict.keys())
        else:
            self.true_path_dict = self.get_true_paths_slow(self.ent2path_df)
            self.true_path_num = len(self.true_path_dict)
            self.true_paths = list(self.true_path_dict.keys())
        self.CLS, self.SEP, self.MASK = n_entity + n_relation, n_entity + n_relation + 1, n_entity + n_relation + 2
        print('Data pre-processing is completed in {} seconds'.format(time()-start))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        center, in_path_data, in_path_num, in_path_weights, \
        out_path_data, out_path_num, out_path_weights = self.ent2path_df.loc[idx, 'center'], \
                                                                         self.ent2path_df.loc[idx, 'ir_path'], \
                                                                         self.ent2path_df.loc[idx, 'i_num'], \
                                                                        self.ent2path_df.loc[idx, 'ir_weight'], \
                                                                         self.ent2path_df.loc[idx, 'or_path'], \
                                                                         self.ent2path_df.loc[idx, 'o_num'], \
                                                                        self.ent2path_df.loc[idx, 'or_weight']
        in_path_array = np.array(in_path_data)
        out_path_array = np.array(out_path_data)

        in_path_set, out_path_set = set([tuple(x) for x in in_path_data]), set([tuple(x) for x in out_path_data])
        CENTER_PATH = np.full((self.samp_size,1), fill_value=center, dtype=np.int64)
        CLS_path, SEP_path = np.full((self.samp_size,1), fill_value=self.CLS, dtype=np.int64), np.full((self.samp_size,1), fill_value=self.SEP, dtype=np.int64)
        # =======Positive samples=============
        in_p_idxs, out_p_idxs = np.random.choice(in_path_num, self.samp_size), np.random.choice(out_path_num, self.samp_size)
        pos_in_path, pos_out_path = in_path_array[in_p_idxs], out_path_array[out_p_idxs]
        # print(pos_in_path.shape, pos_out_path.shape)
        p_paths = np.concatenate([CLS_path, CENTER_PATH, pos_in_path, SEP_path, pos_out_path, SEP_path], axis=1)
        # =======Negative samples=============
        negative_sample_list = []
        negative_sample_size = 0
        round_num = 0
        while negative_sample_size < self.samp_size and round_num < 5:
            negative_idxes = np.random.choice(self.true_path_num, 2*self.samp_size).tolist()
            if self.mode == 'tail_batch':
                negative_samples = [self.true_paths[x] for x in negative_idxes if x not in in_path_set]
            elif self.mode == 'head_batch':
                negative_samples = [self.true_paths[x] for x in negative_idxes if x not in out_path_set]
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_samples = np.array(negative_samples)
            negative_sample_list.append(negative_samples)
            negative_sample_size = negative_sample_size + negative_samples.shape[0]
            round_num = round_num + 1
        neg_samp_paths = np.concatenate(negative_sample_list)[:self.samp_size].reshape(self.samp_size, self.walk_len)
        if self.mode == 'tail_batch':
            n_paths = np.concatenate([CLS_path, CENTER_PATH, neg_samp_paths, SEP_path, pos_out_path, SEP_path], axis=1)
        elif self.mode == 'head_batch':
            n_paths = np.concatenate([CLS_path, CENTER_PATH, pos_in_path, SEP_path, neg_samp_paths, SEP_path], axis=1)
        else:
            raise ValueError('Training batch mode %s not supported' % self.mode)

        paths = np.concatenate([p_paths, n_paths], axis=0)
        labels = np.concatenate([np.ones(self.samp_size, dtype=np.int32), np.zeros(self.samp_size, dtype=np.int32)])
        ent_r_pair, input_pos, input_mask, mask_rel, mask_pos, mask_mask = self.path_position_mask(paths=paths, num_pred=1,
                                                                                              MASK=self.MASK, relation_vocab=self.n_relation)
        paths, paths_pos, paths_mask = torch.LongTensor(ent_r_pair), torch.LongTensor(input_pos), torch.LongTensor(input_mask)
        mask_rel, mask_pos, mask_mask = torch.LongTensor(mask_rel), torch.LongTensor(mask_pos), torch.FloatTensor(mask_mask)
        labels = torch.LongTensor(labels)

        # print('Paths shape {}'.format(paths.shape))
        return paths, paths_pos, paths_mask, mask_rel, mask_pos, mask_mask, labels, self.mode


    def path_position_mask(self, paths, MASK, relation_vocab, num_pred=1):
        m, n = paths.shape[0], paths.shape[1]
        path_pad = paths.copy()
        path_pad[path_pad < 0] = 0
        path_mask = (paths >= 0).astype(np.int16)
        path_pos = np.argsort(-path_mask)
        first_idxs, path_position = np.repeat(np.array(np.arange(m)).reshape(m, 1), n, axis=1), np.repeat(
            np.array(np.arange(n)).reshape(1, n), m, axis=0)
        temp_idx = np.zeros(paths.shape, dtype=np.int16)
        temp_idx[first_idxs, path_pos] = path_position
        path_pos = path_position[first_idxs, temp_idx]
        path_pos[paths < 0] = 0
        # ===========================================
        def get_random_relation(relation_vocab, relation):
            rand_rel = np.random.choice(relation_vocab)
            while rand_rel == relation:
                rand_rel = np.random.choice(relation_vocab)
            return rand_rel

        mask_rels, mask_pos, mask_mask = np.zeros(shape=(m, num_pred), dtype=np.int32), \
                                         np.zeros(shape=(m, num_pred), dtype=np.int32), np.zeros(shape=(m, num_pred),
                                                                                                 dtype=np.int32)
        in_path_rel_mask = ((paths >= 0) & (paths < 11)).astype(np.int16)
        path_lens = in_path_rel_mask.sum(axis=-1)
        for i in range(m):
            if path_lens[i] > num_pred:
                rel_idxes = np.where(in_path_rel_mask[i] == 1)[0]
                mask_pos[i] = rel_idxes[np.random.choice(len(rel_idxes), size=num_pred, replace=False)]
                mask_rels[i] = paths[i, mask_pos[i]]
                mask_mask[i] = 1
                for k in range(num_pred):
                    mask_pos_k = mask_pos[i, k]
                    if rand() < 0.8:
                        path_pad[i, mask_pos_k] = MASK
                    elif rand() < 0.5:
                        path_pad[i, mask_pos_k] = get_random_relation(relation_vocab=relation_vocab,
                                                                      relation=path_pad[i, mask_pos_k])
        # ===========================================
        return path_pad, path_pos, path_mask, mask_rels, mask_pos, mask_mask

    @staticmethod
    def get_ent2path_data_frame(ent2path_data: DataFrame):
        if ent2path_data is None:
            return ent2path_data
        """center: entity, i_num: in number of paths, ir_path: in paths, ir_len: in path len, ir_cnt: frequency of each path, 
        o_num: out number of paths, or_path, or_len, or_cnt"""
        # ent2path = ent2path[(ent2path['i_num'] > 0) | (ent2path['o_num'] > 0)].copy(deep=True)
        ent2path: DataFrame = ent2path_data.copy(deep=True)
        first_ir_path, first_or_path = ent2path.loc[0].at['ir_path'], ent2path.loc[0].at['or_path']
        first_path = first_ir_path if len(first_ir_path) > 0 else first_or_path
        max_path_len = len(first_path[0])
        ent2path.loc[ent2path['i_num'] == 0, 'ir_path'] = [[np.full((1, max_path_len), fill_value=-1, dtype=np.int64)]]
        ent2path.loc[ent2path['o_num'] == 0, 'or_path'] = [[np.full((1, max_path_len), fill_value=-1, dtype=np.int64)]]
        ent2path.loc[ent2path['i_num'] == 0, 'ir_cnt'] = [[np.array([1])]]
        ent2path.loc[ent2path['o_num'] == 0, 'or_cnt'] = [[np.array([1])]]
        ent2path.loc[ent2path['i_num'] == 0, 'ir_len'] = [[0]]
        ent2path.loc[ent2path['o_num'] == 0, 'or_len'] = [[0]]
        ent2path.loc[ent2path['i_num'] == 0, 'i_num'] = 1
        ent2path.loc[ent2path['o_num'] == 0, 'o_num'] = 1
        def weight_computation(row):
            in_weights, out_weights = np.array(row['ir_cnt']),  np.array(row['or_cnt'])
            in_weights, out_weights = in_weights/sum(in_weights), out_weights/sum(out_weights)
            return in_weights, out_weights
        ent2path[['ir_weight', 'or_weight']] = ent2path.apply(weight_computation, axis=1, result_type='expand')
        return ent2path, max_path_len

    @staticmethod
    def get_true_paths(true_paths: DataFrame):
        rel_names = ['r_' + str(i) for i in range(true_paths.shape[1]-1)]
        rel_counts = true_paths['rel_cnt'].tolist()
        rel_paths = true_paths[rel_names].to_numpy()
        rel_paths = [tuple(x) for x in rel_paths]
        true_path_dict = dict(zip(rel_paths, rel_counts))
        return true_path_dict

    @staticmethod
    def collate_fn_path(data):
        # paths = torch.stack([_[0] for _ in data], dim=0)
        # paths_pos = torch.stack([_[1] for _ in data], dim=0)
        # paths_mask = torch.stack([_[2] for _ in data], dim=0)
        # mask_rel = torch.stack([_[3] for _ in data], dim=0)
        # mask_pos = torch.stack([_[4] for _ in data], dim=0)
        # mask_mask = torch.stack([_[5] for _ in data], dim=0)
        # labels = torch.stack([_[6] for _ in data], dim=0)
        paths = torch.cat([_[0] for _ in data], dim=0)
        paths_pos = torch.cat([_[1] for _ in data], dim=0)
        paths_mask = torch.cat([_[2] for _ in data], dim=0)
        mask_rel = torch.cat([_[3] for _ in data], dim=0)
        mask_pos = torch.cat([_[4] for _ in data], dim=0)
        mask_mask = torch.cat([_[5] for _ in data], dim=0)
        labels = torch.cat([_[6] for _ in data], dim=0)
        mode = data[0][7]
        return {'label': labels, 'rel_seq': paths, 'pos_seq': paths_pos, 'mask_seq': paths_mask,
                'm_rel_seq': mask_rel, 'm_pos_seq': mask_pos, 'm_mask_seq': mask_mask, 'mode': mode}

    @staticmethod
    def get_true_paths_slow(ent2path_data: DataFrame):
        true_path_dict = {}
        for idx, row in ent2path_data.iterrows():
            path_data = row['ir_path']
            path_data = [tuple(x) for x in path_data]
            for path in path_data:
                if path in true_path_dict:
                    true_path_dict[path] = true_path_dict[path] + 1
                else:
                    true_path_dict[path] = 1
            path_data = row['or_path']
            path_data = [tuple(x) for x in path_data]
            for path in path_data:
                if path in true_path_dict:
                    true_path_dict[path] = true_path_dict[path] + 1
                else:
                    true_path_dict[path] = 1
        return true_path_dict


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step = self.step + 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        """
        Transform a PyTorch Dataloader into python iterator
        :param dataloader:
        :return:
        """
        while True:
            for data in dataloader:
                yield data

def construct_train_iterator( num_entity, num_relations, sample_size, batch_size, ent2path: DataFrame = None, true_path_df:DataFrame = None,
                             shuffle=True, number_works=8):
    train_dataloader_head = DataLoader(
        BertTrainDataset(n_entity=num_entity, n_relation=num_relations, sample_size=sample_size, mode='head_batch',
                         ent2path_data=ent2path, all_true_paths=true_path_df),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=number_works,
        collate_fn=BertTrainDataset.collate_fn_path)

    train_dataloader_tail=DataLoader(
        BertTrainDataset(n_entity=num_entity, n_relation=num_relations, sample_size= sample_size, mode='tail_batch',
                         ent2path_data=ent2path, all_true_paths=true_path_df),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=number_works,
        collate_fn=BertTrainDataset.collate_fn_path)
    train_iterator = BidirectionalOneShotIterator(dataloader_head=train_dataloader_head, dataloader_tail=train_dataloader_tail)
    return train_iterator

if __name__ == '__main__':
    set_seeds(2019)
    data = dataloader('WN18RR')
    num_nodes = data.num_entities
    num_relations = data.num_relation
    path_pair_save_path = '../SeqData/RelPathPair_WN18RR/'
    path_pair_name = 'RandWalk_walklen_5_epoch_4000_seed_2019_path_pair.json'
    all_true_name = 'RandWalk_walklen_5_epoch_4000_seed_2019_uniq_path.json'

    path_pir_df: DataFrame = load_json_as_data_frame(path_pair_save_path + path_pair_name)
    true_path_df = load_json_as_data_frame(path_pair_save_path + all_true_name)

    # for idx, row in true_path_df.iterrows():
    #     print((row[['r_0', 'r_1']].values))

    # print(path_pir_df.head(1).values)

    # print(true_path_df.values)
    #
    train_iterator = construct_train_iterator(num_entity=num_nodes, num_relations=num_relations, ent2path=path_pir_df, true_path_df=true_path_df,
                                                  sample_size=2, batch_size=100)



    for i in range(2):
        start = time()
        batch = next(train_iterator)
        # for key, value in batch.items():
        #     print(key)
        # batch = {key: value.to('cpu') for key, value in batch.items() if key != 'mode'}
        print(i, time()-start)
        print(batch['rel_seq'].shape, batch['label'].shape)
    # print(time() - start)