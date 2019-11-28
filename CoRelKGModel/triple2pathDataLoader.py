import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


import numpy as np
import torch
from KGEmbedUtils.kgdataUtils import dataloader
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pandas import DataFrame
from time import time
from torch import Tensor
import pandas as pd
from KGEmbedUtils.utils import set_seeds

class TrainDataset(Dataset):
    def __init__(self, triples: list, n_entity: int, n_relation: int, negative_sample_size: int, mode: str, ent2path_data: DataFrame = None):
        self.len =len(triples)
        self.triples = triples
        self.trip_set = set(triples)
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.neg_samp_size = negative_sample_size
        self.mode = mode
        self.true_head, self.true_tail, self.head_rel_count, self.tail_rel_count = self.get_true_head_and_tail(self.triples)
        self.ent2path_df = self.get_ent2path_data_frame(ent2path_data)

    def __len__(self):
        return self.len


    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample
        subsampling_weight = self.head_rel_count[(head, relation)] + self.tail_rel_count[(tail, relation)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.neg_samp_size:
            negative_sample = np.random.randint(self.n_entity, size=self.neg_samp_size * 2)
            if self.mode == 'head_batch':
                mask = np.in1d(negative_sample, self.true_head[(relation, tail)], assume_unique=True, invert=True)
            elif self.mode == 'tail_batch':
                mask = np.in1d(negative_sample, self.true_tail[(head, relation)], assume_unique=True, invert=True)
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size = negative_sample_size + negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.neg_samp_size].reshape(self.neg_samp_size, 1)
        #######
        if self.mode == 'head_batch':
            temp = np.repeat(np.array([[relation, tail]]), self.neg_samp_size, axis=0)
            negative_sample = np.concatenate([negative_sample, temp], axis=1)
        elif self.mode == 'tail_batch':
            temp = np.repeat(np.array([[head, relation]]), self.neg_samp_size, axis=0)
            negative_sample = np.concatenate([temp, negative_sample], axis=1)
        else:
            raise ValueError('Training batch mode %s not supported' % self.mode)
        #######
        negative_sample = torch.from_numpy(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)

        if self.ent2path_df is not None:
            paths, path_mask, path_pos, path_bias_idxs, total_path_num = self.triple2path(positive_sample, negative_sample)
        else:
            raise ValueError('Entity to path data frame is empty' % self.mode)
        return paths, path_mask, path_pos, path_bias_idxs,  total_path_num, subsampling_weight, self.mode

    def triple2path(self, pos_samp, neg_samp):
        CLS, SEP = self.n_relation, self.n_relation + 1
        pn_data = torch.cat([pos_samp.unsqueeze(0), neg_samp], dim=0).numpy()#First is positive/negative
        num_of_triples = pn_data.shape[0]
        #########Generate tree level data#############
        head_data, relations, tail_data = self.ent2path_df.loc[pn_data[:,0], :], \
                                          pn_data[:,1], self.ent2path_df.loc[pn_data[:,2], :]
        in_path_data, in_path_num, out_path_data, out_path_num = head_data['ir_path'].to_numpy(), head_data['i_num'].to_numpy(), \
                                                                 tail_data['or_path'].to_numpy(), tail_data['o_num'].to_numpy()
        # if self.mode == 'tail_batch':
        #     in_path_data, in_path_num, out_path_data, out_path_num = head_data['ir_path'].to_numpy(), head_data['i_num'].to_numpy(), \
        #                                                          tail_data['or_path'].to_numpy(), tail_data['o_num'].to_numpy()
        # elif self.mode == 'head_batch':
        #     in_path_data, in_path_num, out_path_data, out_path_num = tail_data['ir_path'].to_numpy(), tail_data['i_num'].to_numpy(), \
        #                                                          head_data['or_path'].to_numpy(), head_data['o_num'].to_numpy()
        # else:
        #     raise ValueError('negative batch mode %s not supported' % self.mode)

        head_data_array = np.concatenate([np.array(x) for x in in_path_data], axis=0)
        tail_data_array = np.concatenate([np.array(x) for x in out_path_data], axis=0)
        head_num_array, tail_num_array = np.array(in_path_num, dtype=np.int32), np.array(out_path_num, dtype=np.int32)
        #########Generate path level indexes###########
        head_bias, tail_bias = np.add.accumulate(head_num_array), np.add.accumulate(tail_num_array)
        pair_num = head_num_array * tail_num_array
        pair_num_cum = np.add.accumulate(pair_num)
        total_path_num = pair_num.sum()
        head_idxs, tail_idxs, rel_idxs = np.full(total_path_num, fill_value=-1, dtype=np.int64), np.full(total_path_num, fill_value=-1, dtype=np.int64), \
                                          np.full(total_path_num, fill_value=-1, dtype=np.int64)
        head_idxs[:pair_num_cum[0]] = np.repeat(np.array(range(head_num_array[0])), tail_num_array[0])
        tail_idxs[:pair_num_cum[0]] = np.array(list(range(tail_num_array[0])) * head_num_array[0])
        rel_idxs[:pair_num_cum[0]] = 0
        path_num_array = np.array(pair_num)
        for i in range(1, num_of_triples):
            head_idxs[pair_num_cum[i-1]:pair_num_cum[i]] = np.repeat(np.array(range(head_num_array[i])), tail_num_array[i]) + head_bias[i-1]
            tail_idxs[pair_num_cum[i-1]:pair_num_cum[i]] = np.array(list(range(tail_num_array[i])) * head_num_array[i]) + \
                                                     tail_bias[i - 1]
            rel_idxs[pair_num_cum[i-1]:pair_num_cum[i]] = i
        #########Generate path#########################
        head_paths = head_data_array[head_idxs]
        rel_path = relations[rel_idxs]
        tail_paths = tail_data_array[tail_idxs]
        CLS_path, SEP_path = np.full((total_path_num,1), fill_value=CLS, dtype=np.int64), np.full((total_path_num,1), fill_value=SEP, dtype=np.int64)
        paths = np.concatenate([CLS_path, head_paths, rel_path.reshape(total_path_num, 1), tail_paths, SEP_path], axis=1)
        #########Generate path#########################
        #########Generate path#########################
        def path_mask_and_position(paths):
            m, n = paths.shape[0], paths.shape[1]
            path_pad = paths.copy()
            path_pad[path_pad < 0] = 0
            path_mask = (paths >= 0).astype(np.int16)
            path_pos = np.argsort(-path_mask)
            first_idxs, path_position = np.repeat(np.array(np.arange(m)).reshape(m, 1), n, axis=1), np.repeat(np.array(np.arange(n)).reshape(1, n), m, axis=0)
            temp_idx = np.zeros(paths.shape, dtype=np.int16)
            temp_idx[first_idxs, path_pos] = path_position
            path_pos = path_position[first_idxs, temp_idx]
            path_pos[paths < 0] = 0
            return path_pad, path_mask, path_pos
        paths, path_mask, path_pos = path_mask_and_position(paths)
        #########Generate path#########################
        path_bias_idxs = path_num_array
        paths, path_mask, path_pos, path_bias_idxs, total_path_num = torch.from_numpy(paths), torch.from_numpy(path_mask), torch.from_numpy(path_pos), torch.from_numpy(
            path_bias_idxs), torch.LongTensor([total_path_num])
        return paths, path_mask, path_pos, path_bias_idxs, total_path_num

    @staticmethod
    def get_ent2path_data_frame(ent2path_data: DataFrame):
        if ent2path_data is None:
            return ent2path_data
        """center: entity, i_num: in number of paths, ir_path: in paths, ir_len: in path len, ir_cnt: frequency of each path, 
        o_num: out number of paths, or_path, or_len, or_cnt"""
        # ent2path = ent2path[(ent2path['i_num'] > 0) | (ent2path['o_num'] > 0)].copy(deep=True)
        ent2path = ent2path_data.copy(deep=True)
        first_ir_path, first_or_path = ent2path.loc[0].at['ir_path'], ent2path.loc[0].at['or_path']
        first_path = first_ir_path if len(first_ir_path) > 0 else first_or_path
        max_path_len = len(first_path[0])
        ent2path.loc[ent2path['i_num'] == 0, 'ir_path'] = [[np.full((1, max_path_len), fill_value=-1, dtype=np.int64)]]
        ent2path.loc[ent2path['o_num'] == 0, 'or_path'] = [[np.full((1, max_path_len), fill_value=-1, dtype=np.int64)]]
        ent2path.loc[ent2path['i_num'] == 0, 'i_num'] = 1
        ent2path.loc[ent2path['o_num'] == 0, 'o_num'] = 1
        return ent2path

    @staticmethod
    def collate_fn_path(data):
        paths = torch.cat([_[0] for _ in data], dim=0)
        path_mask = torch.cat([_[1] for _ in data], dim=0)
        path_position = torch.cat([_[2] for _ in data], dim=0)
        path_bias_idx = torch.stack([_[3] for _ in data], dim=0)
        path_total_num = torch.stack([_[4] for _ in data], dim=0)
        filter_bias = torch.stack([_[5] for _ in data], dim=0)
        mode = data[0][6]
        # return {'path': paths, 'path_mask': path_mask, 'path_position': path_position,
        #         'path_bias_idx': path_bias_idx, 'path_num': path_total_num,  'batch_weight': filter_bias, 'mode': mode}
        return {'path': paths, 'path_mask': path_mask, 'path_position': path_position,
            'path_bias_idx': path_bias_idx, 'path_num': path_total_num, 'batch_weight': filter_bias}


    @staticmethod
    def get_true_head_and_tail(triples, start=4):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        true_head = {}
        true_tail = {}
        head_rel_count = {}
        tail_rel_count = {}
        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)

            if (head, relation) not in head_rel_count:
                head_rel_count[(head, relation)] = start
            else:
                head_rel_count[(head, relation)] += 1

            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

            if (tail, relation) not in tail_rel_count:
                tail_rel_count[(tail, relation)] = start
            else:
                tail_rel_count[(tail, relation)] += 1

        for relation, tail in true_head.keys():
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail.keys():
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))
        return true_head, true_tail, head_rel_count, tail_rel_count


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


class TestDataset(Dataset):
    def __init__(self, triples: list, all_true_triples:list, nentity:int, nrelation:int, mode:str, epoch_idx:int, chunk_size=100, ent2path_data: DataFrame=None):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode
        self.ent2path_df = self.get_ent2path_data_frame(ent2path_data)
        self.chunk_size = chunk_size
        if self.nentity % chunk_size == 0:
            self.chunk_num = self.nentity // chunk_size
            self.epoch_idx_range = [(i * chunk_size, (i + 1) * chunk_size) for i in range(self.chunk_num)]
        else:
            self.chunk_num = (self.nentity // chunk_size) + 1
            self.epoch_idx_range = [(i * chunk_size, (i + 1) * chunk_size) for i in range(self.chunk_num - 1)]
            self.epoch_idx_range.append((chunk_size * (self.chunk_num - 1), self.nentity))
        self.epoch_idx = epoch_idx

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        if self.mode == 'head_batch':
            tmp = [(0, rand_head, relation, tail) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head, relation, tail) for rand_head in range(self.epoch_idx_range[self.epoch_idx][0], self.epoch_idx_range[self.epoch_idx][1])]
            if head >= self.epoch_idx_range[self.epoch_idx][0] and head < self.epoch_idx_range[self.epoch_idx][1]:
                tmp[head - self.epoch_idx_range[self.epoch_idx][0]] = (0, head, relation, tail)
        elif self.mode == 'tail_batch':
            tmp = [(0, head, relation, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, head, relation, tail) for rand_tail in range(self.epoch_idx_range[self.epoch_idx][0], self.epoch_idx_range[self.epoch_idx][1])]
            if tail >= self.epoch_idx_range[self.epoch_idx][0] and tail < self.epoch_idx_range[self.epoch_idx][1]:
                tmp[tail - self.epoch_idx_range[self.epoch_idx][0]] = (0, head, relation, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)
        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1:]
        positive_sample = torch.LongTensor((head, relation, tail))

        if self.ent2path_df is not None:
            paths, path_mask, path_pos, path_bias_idxs, total_path_num = self.triple2path(positive_sample, negative_sample)
        else:
            raise ValueError('Entity to path data frame is empty')

        return paths, path_mask, path_pos, path_bias_idxs,  total_path_num, filter_bias, self.mode

    def triple2path(self, pos_samp, neg_samp):
        CLS, SEP = self.nrelation, self.nrelation + 1
        pn_data = torch.cat([pos_samp.unsqueeze(0), neg_samp], dim=0).numpy()#First is positive/negative
        num_of_triples = pn_data.shape[0]
        #########Generate tree level data#############
        head_data, relations, tail_data = self.ent2path_df.loc[pn_data[:,0], :], \
                                          pn_data[:,1], self.ent2path_df.loc[pn_data[:,2], :]

        in_path_data, in_path_num, out_path_data, out_path_num = head_data['ir_path'].to_numpy(), head_data[
            'i_num'].to_numpy(), \
                                                                 tail_data['or_path'].to_numpy(), tail_data[
                                                                     'o_num'].to_numpy()
        # if self.mode == 'tail_batch':
        #     in_path_data, in_path_num, out_path_data, out_path_num = head_data['ir_path'].to_numpy(), head_data['i_num'].to_numpy(), \
        #                                                          tail_data['or_path'].to_numpy(), tail_data['o_num'].to_numpy()
        # elif self.mode == 'head_batch':
        #     in_path_data, in_path_num, out_path_data, out_path_num = tail_data['ir_path'].to_numpy(), tail_data['i_num'].to_numpy(), \
        #                                                          head_data['or_path'].to_numpy(), head_data['o_num'].to_numpy()
        # else:
        #     raise ValueError('negative batch mode %s not supported' % self.mode)

        head_data_array = np.concatenate([np.array(x) for x in in_path_data], axis=0)
        tail_data_array = np.concatenate([np.array(x) for x in out_path_data], axis=0)
        head_num_array, tail_num_array = np.array(in_path_num, dtype=np.int32), np.array(out_path_num, dtype=np.int32)
        #########Generate path level indexes###########
        head_bias, tail_bias = np.add.accumulate(head_num_array), np.add.accumulate(tail_num_array)
        pair_num = head_num_array * tail_num_array
        pair_num_cum = np.add.accumulate(pair_num)
        total_path_num = pair_num.sum()
        head_idxs, tail_idxs, rel_idxs = np.full(total_path_num, fill_value=-1, dtype=np.int64), np.full(total_path_num, fill_value=-1, dtype=np.int64), \
                                          np.full(total_path_num, fill_value=-1, dtype=np.int64)
        head_idxs[:pair_num_cum[0]] = np.repeat(np.array(range(head_num_array[0])), tail_num_array[0])
        tail_idxs[:pair_num_cum[0]] = np.array(list(range(tail_num_array[0])) * head_num_array[0])
        rel_idxs[:pair_num_cum[0]] = 0
        path_num_array = np.array(pair_num)
        for i in range(1, num_of_triples):
            head_idxs[pair_num_cum[i-1]:pair_num_cum[i]] = np.repeat(np.array(range(head_num_array[i])), tail_num_array[i]) + head_bias[i-1]
            tail_idxs[pair_num_cum[i-1]:pair_num_cum[i]] = np.array(list(range(tail_num_array[i])) * head_num_array[i]) + \
                                                     tail_bias[i - 1]
            rel_idxs[pair_num_cum[i-1]:pair_num_cum[i]] = i
        #########Generate path#########################
        head_paths = head_data_array[head_idxs]
        rel_path = relations[rel_idxs]
        tail_paths = tail_data_array[tail_idxs]
        CLS_path, SEP_path = np.full((total_path_num,1), fill_value=CLS, dtype=np.int64), np.full((total_path_num,1), fill_value=SEP, dtype=np.int64)
        paths = np.concatenate([CLS_path, head_paths, rel_path.reshape(total_path_num, 1), tail_paths, SEP_path], axis=1)
        #########Generate path#########################
        #########Generate path#########################
        def path_mask_and_position(paths):
            m, n = paths.shape[0], paths.shape[1]
            path_pad = paths.copy()
            path_pad[path_pad < 0] = 0
            path_mask = (paths >= 0).astype(np.int16)
            path_pos = np.argsort(-path_mask)
            first_idxs, path_position = np.repeat(np.array(np.arange(m)).reshape(m, 1), n, axis=1), np.repeat(np.array(np.arange(n)).reshape(1, n), m, axis=0)
            temp_idx = np.zeros(paths.shape, dtype=np.int16)
            temp_idx[first_idxs, path_pos] = path_position
            path_pos = path_position[first_idxs, temp_idx]
            path_pos[paths < 0] = 0
            return path_pad, path_mask, path_pos
        paths, path_mask, path_pos = path_mask_and_position(paths)
        #########Generate path#########################
        path_bias_idxs = path_num_array
        paths, path_mask, path_pos, path_bias_idxs, total_path_num = torch.from_numpy(paths), torch.from_numpy(path_mask), torch.from_numpy(path_pos), torch.from_numpy(
            path_bias_idxs), torch.LongTensor([total_path_num])
        return paths, path_mask, path_pos, path_bias_idxs, total_path_num

    @staticmethod
    def divide_chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    @staticmethod
    def get_ent2path_data_frame(ent2path_data: DataFrame):
        if ent2path_data is None:
            return ent2path_data
        """center: entity, i_num: in number of paths, ir_path: in paths, ir_len: in path len, ir_cnt: frequency of each path, 
        o_num: out number of paths, or_path, or_len, or_cnt"""
        # ent2path = ent2path[(ent2path['i_num'] > 0) | (ent2path['o_num'] > 0)].copy(deep=True)
        ent2path = ent2path_data.copy(deep=True)
        first_ir_path, first_or_path = ent2path.loc[0].at['ir_path'], ent2path.loc[0].at['or_path']
        first_path = first_ir_path if len(first_ir_path) > 0 else first_or_path
        max_path_len = len(first_path[0])
        ent2path.loc[ent2path['i_num'] == 0, 'ir_path'] = [[np.full((1, max_path_len), fill_value=-1, dtype=np.int64)]]
        ent2path.loc[ent2path['o_num'] == 0, 'or_path'] = [[np.full((1, max_path_len), fill_value=-1, dtype=np.int64)]]
        ent2path.loc[ent2path['i_num'] == 0, 'i_num'] = 1
        ent2path.loc[ent2path['o_num'] == 0, 'o_num'] = 1
        return ent2path

    @staticmethod
    def collate_fn_path(data):
        paths = torch.cat([_[0] for _ in data], dim=0)
        path_mask = torch.cat([_[1] for _ in data], dim=0)
        path_position = torch.cat([_[2] for _ in data], dim=0)
        path_bias_idx = torch.stack([_[3] for _ in data], dim=0)
        path_total_num = torch.stack([_[4] for _ in data], dim=0)
        filter_bias = torch.stack([_[5] for _ in data], dim=0)
        mode = data[0][6]
        return {'path': paths, 'path_mask': path_mask, 'path_position': path_position,
                'path_bias_idx': path_bias_idx, 'path_num': path_total_num, 'batch_bias': filter_bias, 'mode': mode}


def construct_test_iterator(test_triples, all_true_triples, num_entity, num_relations, epoch_idx, ent2path: DataFrame=None, chunk_size=200, batch_size=2, number_workers=8):
    epoch_num = num_entity // chunk_size if num_entity % chunk_size == 0 else (num_entity // chunk_size + 1)
    if epoch_idx >= epoch_num:
        raise ValueError('Out of epoch index {} >= {}'.format(epoch_idx, epoch_num))
    test_dataloader_head = DataLoader(
        TestDataset(triples=test_triples,all_true_triples=all_true_triples, nentity=num_entity, epoch_idx=epoch_idx, chunk_size=chunk_size, nrelation=num_relations, mode='head_batch', ent2path_data=ent2path),
        batch_size=batch_size,
        num_workers=number_workers,
        shuffle=False,
        collate_fn=TestDataset.collate_fn_path)

    test_dataloader_tail = DataLoader(
        TestDataset(triples=test_triples,all_true_triples=all_true_triples, nentity=num_entity, epoch_idx=epoch_idx, chunk_size=chunk_size, nrelation=num_relations, mode='tail_batch', ent2path_data=ent2path),
        batch_size=batch_size,
        num_workers=number_workers,
        shuffle=False,
        collate_fn=TestDataset.collate_fn_path)
    return [test_dataloader_head, test_dataloader_tail]

def construct_train_iterator(train_triples, num_entity, num_relations, negative_sample_size, batch_size, ent2path: DataFrame = None, shuffle=True, number_works=8):
    train_dataloader_head = DataLoader(
        TrainDataset(train_triples, num_entity, num_relations, negative_sample_size, 'head_batch', ent2path_data=ent2path),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=number_works,
        collate_fn=TrainDataset.collate_fn_path)

    train_dataloader_tail=DataLoader(
        TrainDataset(train_triples, num_entity, num_relations, negative_sample_size, 'tail_batch', ent2path_data=ent2path),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=number_works,
        collate_fn=TrainDataset.collate_fn_path)
    train_iterator = BidirectionalOneShotIterator(dataloader_head=train_dataloader_head, dataloader_tail=train_dataloader_tail)
    return train_iterator


def data_split(data: Tensor, path_tri_nums: Tensor):
    batch_size, neg_samp_size = path_tri_nums.shape[0], path_tri_nums.shape[1] - 1
    trip_split_sizes = tuple(path_tri_nums.view(batch_size * (neg_samp_size + 1)).tolist())
    res = torch.split(data, split_size_or_sections=trip_split_sizes, dim=0)
    res_pool = torch.stack([avergage_pool(x.float()) for x in res])
    print(res_pool.shape)
    return res, batch_size, neg_samp_size

def avergage_pool(data: Tensor):
    return torch.mean(data, dim=0)

def max_pool(data: Tensor):
    return torch.max(data, dim=0)[0]

def min_pool(data: Tensor):
    return torch.min(data, dim=0)[0]


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

if __name__ == '__main__':
    import networkx as nx
    from KGEmbedUtils.ioutils import load_json_as_data_frame

    x = divide_chunks(33, 10)
    for y in x:
        print(y)
    # print(x)

#     set_seeds(2019)
#     entity_save_path = '../SeqData/Entity2PathWN18RR/'
#     # entity_save_path = '../SeqData/Entity2PathFB_15K_237/'
#     # data = dataloader('FB15k-237')
#     data = dataloader('WN18RR')
#     num_nodes = data.num_entities
#     num_relations = data.num_relation
#     train_data = data.train
#     relations = data.rel_dict_rev
#     print("Number of entities = {}\nNumber of Edges = {}\nNumber of relations = {}.".format(num_nodes,
#                                                                                             train_data.shape[0], num_relations))
#     valid_data = data.valid
#     test_data = data.test
#     print('Train: {}, valid: {}, test: {}'.format(train_data.shape[0], valid_data.shape[0], test_data.shape[0]))
#     all_data = np.concatenate([train_data, valid_data, test_data], axis=0)
#     all_true_triples = [tuple(x) for x in all_data.tolist()]
#     test_triples = [tuple(x) for x in test_data.tolist()]
# #
# #     # start = time()
#     ent2path = load_json_as_data_frame(entity_save_path + 'kg_entity_to_tree_path_hop_3.json')
#     # test_datat_list = construct_test_iterator(test_triples=test_triples, all_true_triples=all_true_triples, num_entity=num_nodes, chunk_size=200, epoch_idx=0, batch_size=1,
#     #                                           num_relations=num_relations, ent2path=ent2path)
#     #
#     # start = time()
#     # idx = 0
#     # for batch in test_datat_list[0]:
#     #     print(batch['path'].shape)
#     #     print(idx, time() - start)
#     #     idx = idx + 1
#
#     # train_triples = [tuple(x) for x in train_data.tolist()]
#     # train_iterator = construct_train_iterator(train_triples=train_triples, num_entity=num_nodes, num_relations=num_relations, ent2path=ent2path,
#     #                                           negative_sample_size=31, batch_size=4)
#     # # #=================================================================
#     # #
#     # #
#     # # # print(data[''])
#     # # # #=================================================================
#     # start = time()
#     # for i in range(1000):
#     #     batch = next(train_iterator)
#     #     print(batch['path'].shape)
#     #     # paths, path_mask, path_position, path_batch_num, path_total_num, weight = batch['path'], batch[
#     #     #     'path_mask'], \
#     #     #                                                                                 batch['path_position'], batch[
#     #     #                                                                                     'path_bias_idx'], \
#     #     #                                                                                 batch['path_num'], batch[
#     #     #                                                                                     'batch_weight']
#     #     # data_split(paths, path_tri_nums=path_batch_num)
#     #
#     #
#     #     # print(batch_i['path'].shape)
#     #     # print(pos.shape, neg.shape)
#     #     print(i, time() - start)
#
