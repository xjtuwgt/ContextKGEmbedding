from pandas import DataFrame
import pandas as pd
import numpy as np
from KGEmbedUtils.utils import set_seeds
from KGEmbedUtils.ioutils import load_json_as_data_frame
import torch
from time import time

class RelationPathPairDataLoader(object):
    def __init__(self, data_frame: DataFrame, batch_size=32, shuffle=True, drop_last=False):
        """
        :param data_frame:
        :param batch_size:
        :param shuffle:
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cur_idx = 0
        self.cur_batch_idx = 0
        self.pd_dataframe = data_frame
        self.data_size = self.pd_dataframe.shape[0]
        self.drop_last = drop_last
        if self.data_size % self.batch_size == 0:
            self.num_batch = (self.data_size // self.batch_size)
        else:
            if self.drop_last:
                self.num_batch = (self.data_size // self.batch_size)
            else:
                self.num_batch = (self.data_size // self.batch_size) + 1

        if self.shuffle:
            self.data_shuffle()

    def data_shuffle(self):
        self.pd_dataframe = self.pd_dataframe.reindex(np.random.permutation(self.pd_dataframe.index))
        self.batch_reset()

    def batch_reset(self):
        self.cur_idx = 0
        self.cur_batch_idx = 0

    def next_batch(self):
        if self.cur_idx < self.data_size and (self.cur_idx + self.batch_size < self.data_size or self.data_size % self.batch_size == 0):
            this_batch_df = self.pd_dataframe.iloc[self.cur_idx:(self.cur_idx + self.batch_size)]
        elif self.cur_idx < self.data_size and self.cur_idx + self.batch_size > self.data_size:
            this_batch_df = self.pd_dataframe.iloc[self.cur_idx:]
        else:
            raise IndexError
        self.cur_idx = self.cur_idx + self.batch_size
        self.cur_batch_idx = self.cur_batch_idx + 1
        return pd.DataFrame(this_batch_df)

    def has_next_batch(self):
        return self.cur_batch_idx < self.num_batch

    def next_batch_tensor(self):
        data = self.next_batch()
        'label r_pair pos_pair input_mask mask_rel mask_pos mask_mask'
        label = torch.LongTensor(data['label'].tolist()) # 4-class
        label[label > 0] = 1 #binary class
        rel_seq = torch.LongTensor(data['r_pair'].tolist())
        pos_seq = torch.LongTensor(data['pos_pair'].tolist())
        mask_seq = torch.ByteTensor(data['input_mask'].tolist())
        m_rel_seq = torch.LongTensor(data['mask_rel'].tolist())
        m_pos_seq = torch.LongTensor(data['mask_pos'].tolist())
        m_mask_seq = torch.FloatTensor(data['mask_mask'].tolist())
        return {'label': label, 'rel_seq': rel_seq, 'pos_seq': pos_seq,
                'mask_seq': mask_seq, 'm_rel_seq': m_rel_seq, 'm_pos_seq': m_pos_seq, 'm_mask_seq': m_mask_seq}

if __name__ == '__main__':
    pair_data_path = '../SeqData/RelPathPair_WN18RR/'
    seed = 2019
    epochs = 200
    walk_len = 8
    samp_type = 'weighted'
    # samp_type = 'uniform'
    num_of_pairs = 100000
    set_seeds(seed)
    relation_num = 11

    rel_path_pair_data_file_masked_name = 'Masked_Relation_Walk_Epoch_' + str(epochs) + "_walklen_" + str(
        walk_len) + '_seed_' + str(seed) + '_pairnum_' + str(num_of_pairs) + '_samp_type_' + samp_type

    data = load_json_as_data_frame(pair_data_path + rel_path_pair_data_file_masked_name +'.json')
    print(data.shape)
    batch_data = RelationPathPairDataLoader(data_frame=data, batch_size=4)
    # for col in data.columns:
    #     print(col)
    # print(batch_data.num_batch)
    idx = 0
    start_0 = time()
    # x = torch.rand((4,17,5))

    while batch_data.has_next_batch():
        start = time()
        batch_i = batch_data.next_batch_tensor()
        masked_pos = batch_i['rel_seq']
        print(masked_pos.shape)
        # print(masked_pos[:,:,None].expand(-1,-1,5).shape)
        # mm_pos = masked_pos[:,:,None].expand(-1,-1,5)
        # print(masked_pos)
        # print(mm_pos)
        # print(mm_pos.shape)
        # x_mask = torch.gather(x, 1, mm_pos)
        #
        # # print(x.shape)
        # # print(x_mask.shape)

        print(time() - start, idx)
        # print(batch_i.shape, idx, batch_data.num_batch)
        idx = idx + 1
    print(time() - start_0)