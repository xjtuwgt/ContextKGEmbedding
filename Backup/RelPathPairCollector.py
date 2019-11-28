from pandas import DataFrame
import pandas as pd
from KGEmbedUtils.ioutils import load_json_as_data_frame, save_to_json
from time import time
import random
import bisect
from KGEmbedUtils.utils import set_seeds
from pandarallel import pandarallel
import numpy as np
import multiprocessing
num_works = multiprocessing.cpu_count()
if num_works > 16:
    num_works = 16
pandarallel.initialize(progress_bar=True, nb_workers=num_works)

#=============================================================
class WeightedRandomGenerator(object):
    def __init__(self, weights):
        self.totals = []
        running_total = 0
        for w in weights:
            running_total += w
            self.totals.append(running_total)

    def next(self):
        rnd = random.random() * self.totals[-1]
        return bisect.bisect_right(self.totals, rnd)

    def __call__(self):
        return self.next()

def label_row(row):
    first_pair_set, second_pair_set = row['pair_set1'], row['pair_set2']
    first_head_set, second_head_set = row['head_set1'], row['head_set2']
    first_tail_set, second_tail_set = row['tail_set1'], row['tail_set2']
    def intersection_set(first_set: set, second_set: set):
        return len(first_set.intersection(second_set)) > 0
    def equal_set(first_set: set, second_set: set):
        return len(first_set.difference(second_set)) == 0
    def pair_label(first_pair_set, second_pair_set, first_head_set, second_head_set, first_tail_set, second_tail_set):
        if intersection_set(first_pair_set, second_pair_set):
            if equal_set(first_pair_set, second_pair_set):
                return 3
            else:
                return 0
        elif intersection_set(first_head_set, second_head_set):
            return 1
        elif intersection_set(first_tail_set, second_tail_set):
            return 2
        else:
            return 3
    label = pair_label(first_pair_set, second_pair_set, first_head_set, second_head_set, first_tail_set,
                       second_tail_set)
    return label
#=============================================================
class relation_path_pair_generater(object):
    def __init__(self, json_relation_pair_file_name, samp_type: str):
        self.data, self.weights, self.population = self.pos_neg_data_frame(json_relation_pair_file_name)
        self.pos_sampler = WeightedRandomGenerator(weights=self.weights)
        self.samp_type = samp_type

    def pos_neg_data_frame(self, json_file_name) -> (DataFrame, list, list):
        start = time()
        data = load_json_as_data_frame(json_file_name)
        weights = data['p_cnt'].to_numpy() ## Random Walk Count
        weights = weights/weights.sum()
        def dict_colection(row):
            pair_dict = dict()
            head_dict = dict()
            tail_dict = dict()
            triple_list = row['p_pir']
            for trip in triple_list:
                head, tail, count = trip[0], trip[1], trip[2]
                if (head, tail) not in pair_dict:
                    pair_dict[(head, tail)] = count
                else:
                    pre_count = pair_dict[(head, tail)]
                    pair_dict[(head, tail)] = pre_count + count
                if head not in head_dict:
                    head_dict[head] = count
                else:
                    pre_count = head_dict[head]
                    head_dict[head] = pre_count + count
                if tail not in tail_dict:
                    tail_dict[tail] = count
                else:
                    pre_count = tail_dict[tail]
                    tail_dict[tail] = pre_count + count
            return pair_dict, head_dict, tail_dict, set(pair_dict.keys()), set(head_dict.keys()), set(tail_dict.keys())
        data[['pair_dict', 'head_dict', 'tail_dict', 'pair_set', 'head_set', 'tail_set']] = data.apply(dict_colection, axis=1, result_type="expand")
        idxes = list(range(data.shape[0]))
        print('Sampling weights are generated in {:.2f} seconds'.format(time() - start))
        return (data, weights, idxes)

    def relation_pair_generation_fast(self, num_pairs):
        start = time()
        first_idx_list, second_idx_list = self.index_pair_generation(num_pairs=num_pairs)
        rel_names_1 = dict([(rel, rel + str(1)) for rel in self.data.columns])
        rel_names_2 = dict([(rel, rel + str(2)) for rel in self.data.columns])
        first_df, second_df = self.data.iloc[first_idx_list], self.data.iloc[second_idx_list]
        first_df = first_df.rename(columns=rel_names_1).reset_index(drop=True)
        second_df = second_df.rename(columns=rel_names_2).reset_index(drop=True)
        relation_pairs = pd.concat([first_df, second_df], axis=1)
        relation_pairs['label'] = relation_pairs.parallel_apply(label_row, axis=1)
        relation_names_drop = [col for col in relation_pairs.columns if col not in ['label', 'p1', 'p2', 'p_len1', 'p_len2']]
        relation_pairs = relation_pairs.drop(columns=relation_names_drop)
        print('Generate {} pairs in {:.2f} seconds'.format(num_pairs, time() - start))
        return relation_pairs

    def index_pair_generation_weighted_samp(self, num_pairs):
        first_idx_list = np.zeros(num_pairs, dtype=np.int64)
        second_idx_list = np.zeros(num_pairs, dtype=np.int64)
        for i in range(num_pairs):
            idx_first = self.population[self.pos_sampler.next()]
            idx_second = self.population[self.pos_sampler.next()]
            while idx_second == idx_first:
                idx_second = self.population[self.pos_sampler.next()]
            first_idx_list[i] = idx_first
            second_idx_list[i] = idx_second
        return first_idx_list, second_idx_list

    def index_pair_generation_random_samp(self, num_pairs):
        first_idx_list = np.zeros(num_pairs, dtype=np.int64)
        second_idx_list = np.zeros(num_pairs, dtype=np.int64)
        for i in range(num_pairs):
            idx_first = random.choice(self.population)
            idx_second = random.choice(self.population)
            while idx_second == idx_first:
                idx_second = random.choice(self.population)
            first_idx_list[i] = idx_first
            second_idx_list[i] = idx_second
        return first_idx_list, second_idx_list

    def index_pair_generation_reservoir_samp(self, num_pairs):
        first_idx_list = np.zeros(num_pairs, dtype=np.int64)
        second_idx_list = np.zeros(num_pairs, dtype=np.int64)
        population_size = len(self.population)
        total_num = int((population_size * population_size - population_size)/2)
        print(total_num, population_size)
        temp_samples = np.random.choice(total_num, num_pairs, replace=False)
        temp_cum_sum = np.cumsum(np.arange(population_size))
        for i in range(num_pairs):
            temp_sam = temp_samples[i]
            first_idx = np.searchsorted(temp_cum_sum, temp_sam, side='right')
            second_idx = temp_sam - temp_cum_sum[first_idx - 1]
            first_idx_list[i] = first_idx
            second_idx_list[i] = second_idx
            # print(first_idx, second_idx)
        return first_idx_list, second_idx_list

    def index_pair_generation(self, num_pairs):
        if self.samp_type == 'weighted':
            return self.index_pair_generation_weighted_samp(num_pairs)
        if self.samp_type == 'uniform':
            return self.index_pair_generation_random_samp(num_pairs)
        if self.samp_type == 'reservoir':
            return self.index_pair_generation_reservoir_samp(num_pairs)

if __name__ == '__main__':
    # import numpy as np
    seed = 2019
    epochs = 4000
    walk_len = 8
    # samp_type = 'weighted'
    samp_type = 'uniform'
    num_of_pairs = 5000000
    set_seeds(seed)

    seq_data_path = '../SeqData/RandWalk_WN18RR/'
    pair_data_path = '../SeqData/RelPathPair_WN18RR/'
    rel_pair_data_file_name = 'Relation_Walk_Epoch_' + str(epochs) + "_walklen_" + str(walk_len) + '_seed_' + str(seed)
    # df = load_json_as_data_frame(seq_data_path + rel_pair_data_file_name + ".json")##p: path, p_len: path_len, p_cnt: p_count, p_pir: p_head_tail_entity
    reader = relation_path_pair_generater(json_relation_pair_file_name=seq_data_path + rel_pair_data_file_name + ".json", samp_type=samp_type)

    start = time()
    path_pair_data = reader.relation_pair_generation_fast(num_pairs=num_of_pairs)
    print('runtime = {}'.format(time() - start))
    print(path_pair_data['label'].value_counts(normalize=True, sort=True))
    # print(path_pair_data)
    # rel_path_pair_data_file_name = 'Relation_Walk_Epoch_' + str(epochs) + "_walklen_" + str(walk_len) + '_seed_' + str(seed) + '_pairnum_' + str(num_of_pairs) + '_samp_type_' + samp_type
    # save_to_json(data=path_pair_data, file_name=pair_data_path + rel_path_pair_data_file_name + '.json')
    # df = load_json_as_data_frame(pair_data_path + rel_path_pair_data_file_name + '.json')
    # print(df['label'].value_counts(normalize=True, sort=True))