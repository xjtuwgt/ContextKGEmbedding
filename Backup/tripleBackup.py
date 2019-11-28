import numpy as np
import torch
from KGEmbedUtils.kgdataUtils import dataloader
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pandas import DataFrame
from time import time
import pandas as pd

class TrainDataset(Dataset):
    def __init__(self, triples: list, n_entity: int, n_relation: int, negative_sample_size: int, mode):
        self.len =len(triples)
        self.triples = triples
        self.trip_set = set(triples)
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.neg_samp_size = negative_sample_size
        self.mode = mode
        self.true_head, self.true_tail, self.head_rel_count, self.tail_rel_count = self.get_true_head_and_tail(self.triples)

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

        return positive_sample, negative_sample, subsampling_weight, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        print(positive_sample.shape)
        return positive_sample, negative_sample, filter_bias, mode

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
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        if self.mode == 'head_batch':
            tmp = [(0, rand_head, relation, tail) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head, relation, tail) for rand_head in range(self.nentity)]
            tmp[head] = (0, head, relation, tail)
        elif self.mode == 'tail_batch':
            tmp = [(0, head, relation, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, head, relation, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, head, relation, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)
        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1:]
        positive_sample = torch.LongTensor((head, relation, tail))
        return positive_sample, negative_sample, filter_bias, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode

def test_batch_to_paths(batch, data: DataFrame, num_relation:int, npartition=16):
    CLS, SEP = num_relation, num_relation + 1
    pos, neg, weight, mode = batch
    batch_size, negative_size = neg.shape[0], neg.shape[1]
    neg_data = neg.view(batch_size * negative_size, -1)
    pn_data = torch.cat([pos, neg_data], dim=0)
    flag = torch.tensor([0]*batch_size + [1]*(batch_size * negative_size), dtype=torch.int64).view((batch_size * negative_size + batch_size, 1))
    batch_no = torch.tensor(list(range(batch_size)) + list(range(negative_size)) * batch_size).view((batch_size * negative_size + batch_size, 1))
    batch_data = torch.cat([batch_no, flag, pn_data], dim=1).numpy()
    batch_df = pd.DataFrame(batch_data, columns=['b_no', 'flag', 'h', 'r', 't'])

    def path_mask_and_position(paths, path_lens):
        n, m = paths.shape[0], paths.shape[1]
        positions, path_mask, path_pad = np.zeros(paths.shape, dtype=np.int32), \
                                         np.zeros(paths.shape, dtype=np.int16), \
                                         np.zeros(paths.shape, dtype=np.int16)
        bool_paths = paths >= 0
        for i in range(n):
            path_pad[i, :path_lens[i]] = paths[i, bool_paths[i]]
            path_mask[i, :path_lens[i]] = 1
            positions[i, :path_lens[i]] = list(range(path_lens[i]))
        return path_pad, path_mask, positions

    def triple2path(row):
        h, r, t = row['h'], row['r'], row['t']
        head_ir_paths, head_ir_path_len, head_or_paths, head_or_path_len = data.loc[h, 'ir_path'], data.loc[
            h, 'ir_len'], \
                                                                           data.loc[h, 'or_path'], data.loc[h, 'or_len']
        tail_ir_paths, tail_ir_path_len, tail_or_paths, tail_or_path_len = data.loc[t, 'ir_path'], data.loc[
            t, 'ir_len'], \
                                                                           data.loc[t, 'or_path'], data.loc[t, 'or_len']

        def combination(head, rel, tail, head_len, tail_len):
            if len(head) == 0 and len(tail) == 0:
                return None, None
            if len(head) == 0 and len(tail) > 0:
                tail = np.array(tail, dtype=np.int64)
                tail_size = tail.shape[0]
                head_array = np.full((tail.shape[0], tail.shape[1]), fill_value=-1, dtype=np.int64)
                rel_array, CLS_array, SEP_array = np.full((tail_size, 1), fill_value=rel, dtype=np.int64), \
                                                  np.full((tail_size, 1), fill_value=CLS, dtype=np.int64), \
                                                  np.full((tail_size, 1), fill_value=SEP, dtype=np.int64)
                paths = np.concatenate([CLS_array, head_array, rel_array, tail, SEP_array], axis=1)
                path_lens = np.array(tail_len, dtype=np.int16) + 3
                return paths, path_lens
            if len(head) > 0 and len(tail) == 0:
                head = np.array(head, dtype=np.int64)
                head_size = head.shape[0]
                tail_array = np.full((head.shape[0], head.shape[1]), fill_value=-1, dtype=np.int64)
                rel_array, CLS_array, SEP_array = np.full((head_size, 1), fill_value=rel, dtype=np.int64), \
                                                  np.full((head_size, 1), fill_value=CLS, dtype=np.int64), \
                                                  np.full((head_size, 1), fill_value=SEP, dtype=np.int64)
                paths = np.concatenate([CLS_array, head, rel_array, tail_array, SEP_array], axis=1)
                path_lens = np.array(head_len, dtype=np.int16) + 3
                return paths, path_lens
            head_array, tail_array = np.array(head, dtype=np.int64), np.array(tail, dtype=np.int64)
            tail_len_array, head_len_array = np.array(tail_len, dtype=np.int16), np.array(head_len, dtype=np.int16)
            head_size, tail_size = head_array.shape[0], tail_array.shape[0]
            head_idxes = np.repeat(np.array(range(head_size)), tail_size)
            tail_idxes = np.array(list(range(tail_size)) * head_size)
            head_array, tail_array = head_array[head_idxes], tail_array[tail_idxes]
            rel_array, CLS_array, SEP_array = np.full((head_size * tail_size, 1), fill_value=rel, dtype=np.int64), \
                                              np.full((head_size * tail_size, 1), fill_value=CLS, dtype=np.int64), \
                                              np.full((head_size * tail_size, 1), fill_value=SEP, dtype=np.int64)
            paths = np.concatenate([CLS_array, head_array, rel_array, tail_array, SEP_array], axis=1)
            path_lens = tail_len_array[tail_idxes] + head_len_array[head_idxes] + 3
            return paths, path_lens
        h2t_paths, h2t_path_lens = combination(head_ir_paths, r, tail_or_paths, head_ir_path_len, tail_or_path_len)
        t2h_paths, t2h_path_lens = combination(tail_ir_paths, r, head_or_paths, tail_ir_path_len, head_or_path_len)
        if h2t_paths is None and t2h_paths is None:
            paths, path_lens, split_idx = None, None, 0
        elif h2t_paths is None and t2h_paths is not None:
            paths, path_lens, split_idx = t2h_paths, t2h_path_lens, len(t2h_path_lens)
        elif t2h_paths is None and h2t_paths is not None:
            paths, path_lens, split_idx = h2t_paths, h2t_path_lens, len(h2t_path_lens)
        else:
            paths = np.concatenate([h2t_paths, t2h_paths], axis=0)
            path_lens = np.concatenate([h2t_path_lens, t2h_path_lens])
            split_idx = len(h2t_paths)

        path_num = 0
        if paths is not None:
            paths, path_mask, path_positions = path_mask_and_position(paths, path_lens)
            path_num = paths.shape[0]
        else:
            paths, path_mask, path_positions = None, None, None
        return paths, path_mask, path_positions, split_idx, path_num
    batch_df[['t_p', 't_p_m', 't_p_pos', 't_split', 'p_n']] = batch_df.apply(triple2path, axis=1, result_type='expand')
    # print(batch_df.shape, batch_df['p_n'].to_numpy().sum())
    return batch_df

def train_batch_to_paths(batch, data: DataFrame, num_relation:int):
    CLS, SEP = num_relation, num_relation + 1
    pos, neg, weight, mode = batch
    batch_size, negative_size = neg.shape[0], neg.shape[1]
    neg_data = neg.view(batch_size * negative_size, -1)
    pn_data = torch.cat([pos, neg_data], dim=0)
    flag = torch.tensor([0]*batch_size + [1]*(batch_size * negative_size), dtype=torch.int64).view((batch_size * negative_size + batch_size, 1))
    batch_idx = torch.tensor(list(range(batch_size)) + list(range(negative_size)) * batch_size).view((batch_size * negative_size + batch_size, 1))
    batch_data = torch.cat([batch_idx, flag, pn_data], dim=1).numpy()
    batch_df = pd.DataFrame(batch_data, columns=['b_no', 'flag', 'h', 'r', 't'])

    def path_mask_and_position(paths, path_lens):
        n, m = paths.shape[0], paths.shape[1]
        positions, path_mask, path_pad = np.zeros(paths.shape, dtype=np.int32), \
                                         np.zeros(paths.shape, dtype=np.int16), \
                                         np.zeros(paths.shape, dtype=np.int16)
        bool_paths = paths >= 0
        for i in range(n):
            path_pad[i, :path_lens[i]] = paths[i, bool_paths[i]]
            path_mask[i, :path_lens[i]] = 1
            positions[i, :path_lens[i]] = list(range(path_lens[i]))
        return path_pad, path_mask, positions

    def triple2path(row):
        h, r, t = row['h'], row['r'], row['t']
        head_ir_paths, head_ir_path_len, head_or_paths, head_or_path_len = data.loc[h, 'ir_path'], data.loc[
            h, 'ir_len'], \
                                                                           data.loc[h, 'or_path'], data.loc[h, 'or_len']
        tail_ir_paths, tail_ir_path_len, tail_or_paths, tail_or_path_len = data.loc[t, 'ir_path'], data.loc[
            t, 'ir_len'], \
                                                                           data.loc[t, 'or_path'], data.loc[t, 'or_len']

        def combination(head, rel, tail, head_len, tail_len):
            if len(head) == 0 and len(tail) == 0:
                return None, None
            if len(head) == 0 and len(tail) > 0:
                tail = np.array(tail, dtype=np.int64)
                tail_size = tail.shape[0]
                head_array = np.full((tail.shape[0], tail.shape[1]), fill_value=-1, dtype=np.int64)
                rel_array, CLS_array, SEP_array = np.full((tail_size, 1), fill_value=rel, dtype=np.int64), \
                                                  np.full((tail_size, 1), fill_value=CLS, dtype=np.int64), \
                                                  np.full((tail_size, 1), fill_value=SEP, dtype=np.int64)
                paths = np.concatenate([CLS_array, head_array, rel_array, tail, SEP_array], axis=1)
                path_lens = np.array(tail_len, dtype=np.int16) + 3
                return paths, path_lens
            if len(head) > 0 and len(tail) == 0:
                head = np.array(head, dtype=np.int64)
                head_size = head.shape[0]
                tail_array = np.full((head.shape[0], head.shape[1]), fill_value=-1, dtype=np.int64)
                rel_array, CLS_array, SEP_array = np.full((head_size, 1), fill_value=rel, dtype=np.int64), \
                                                  np.full((head_size, 1), fill_value=CLS, dtype=np.int64), \
                                                  np.full((head_size, 1), fill_value=SEP, dtype=np.int64)
                paths = np.concatenate([CLS_array, head, rel_array, tail_array, SEP_array], axis=1)
                path_lens = np.array(head_len, dtype=np.int16) + 3
                return paths, path_lens
            head_array, tail_array = np.array(head, dtype=np.int64), np.array(tail, dtype=np.int64)
            tail_len_array, head_len_array = np.array(tail_len, dtype=np.int16), np.array(head_len, dtype=np.int16)
            head_size, tail_size = head_array.shape[0], tail_array.shape[0]
            head_idxes = np.repeat(np.array(range(head_size)), tail_size)
            tail_idxes = np.array(list(range(tail_size)) * head_size)
            head_array, tail_array = head_array[head_idxes], tail_array[tail_idxes]
            rel_array, CLS_array, SEP_array = np.full((head_size * tail_size, 1), fill_value=rel, dtype=np.int64), \
                                              np.full((head_size * tail_size, 1), fill_value=CLS, dtype=np.int64), \
                                              np.full((head_size * tail_size, 1), fill_value=SEP, dtype=np.int64)
            paths = np.concatenate([CLS_array, head_array, rel_array, tail_array, SEP_array], axis=1)
            path_lens = tail_len_array[tail_idxes] + head_len_array[head_idxes] + 3
            return paths, path_lens
        h2t_paths, h2t_path_lens = combination(head_ir_paths, r, tail_or_paths, head_ir_path_len, tail_or_path_len)
        t2h_paths, t2h_path_lens = combination(tail_ir_paths, r, head_or_paths, tail_ir_path_len, head_or_path_len)
        if h2t_paths is None and t2h_paths is None:
            paths, path_lens, split_idx = None, None, 0
        elif h2t_paths is None and t2h_paths is not None:
            paths, path_lens, split_idx = t2h_paths, t2h_path_lens, len(t2h_path_lens)
        elif t2h_paths is None and h2t_paths is not None:
            paths, path_lens, split_idx = h2t_paths, h2t_path_lens, len(h2t_path_lens)
        else:
            paths = np.concatenate([h2t_paths, t2h_paths], axis=0)
            path_lens = np.concatenate([h2t_path_lens, t2h_path_lens])
            split_idx = len(h2t_paths)

        path_num = 0
        if paths is not None:
            paths, path_mask, path_positions = path_mask_and_position(paths, path_lens)
            path_num = paths.shape[0]
        else:
            paths, path_mask, path_positions = None, None, None
        return paths, path_mask, path_positions, split_idx, path_num
    batch_df[['t_p', 't_p_m', 't_p_pos', 't_split', 'p_n']] = batch_df.apply(triple2path, axis=1, result_type='expand')
    # print(batch_df.shape, batch_df['p_n'].to_numpy().sum())
    return batch_df



def construct_test_iterator(test_triples, all_true_triples, num_entity, num_relations, batch_size=2, number_workers=4):
    test_dataloader_head = DataLoader(
        TestDataset(triples=test_triples,all_true_triples=all_true_triples, nentity=num_entity, nrelation=num_relations, mode='head_batch'),
        batch_size=batch_size,
        num_workers=number_workers,
        collate_fn=TestDataset.collate_fn)

    test_dataloader_tail = DataLoader(
        TestDataset(triples=test_triples,all_true_triples=all_true_triples, nentity=num_entity, nrelation=num_relations, mode='tail_batch'),
        batch_size=batch_size,
        num_workers=number_workers,
        collate_fn=TestDataset.collate_fn)
    return [test_dataloader_head, test_dataloader_tail]

def construct_train_iterator(train_triples, num_entity, num_relations, negative_sample_size, batch_size, shuffle=True, number_works=4):
    train_dataloader_head = DataLoader(
        TrainDataset(train_triples, num_entity, num_relations, negative_sample_size, 'head_batch'),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=number_works,
        collate_fn=TrainDataset.collate_fn)

    train_dataloader_tail=DataLoader(
        TrainDataset(train_triples, num_entity, num_relations, negative_sample_size, 'tail_batch'),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=number_works,
        collate_fn=TrainDataset.collate_fn)
    train_iterator = BidirectionalOneShotIterator(dataloader_head=train_dataloader_head, dataloader_tail=train_dataloader_tail)
    return train_iterator