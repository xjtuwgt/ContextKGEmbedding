import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from random import shuffle
from random import random as rand
import random
from pandas import DataFrame
from KGEmbedUtils.ioutils import load_json_as_data_frame, save_to_json
import numpy as np
from KGEmbedUtils.utils import set_seeds
from time import time
from pandarallel import pandarallel
import multiprocessing
num_works = multiprocessing.cpu_count()
if num_works > 16:
    num_works = 16
pandarallel.initialize(progress_bar=True, nb_workers=num_works)

def mask_relation_pair_data(data: DataFrame,  num_relation: int, max_pred_num: int = 3, mask_prob: float = 0.15,  max_len=17):
    CLS, SEP, MASK = num_relation, num_relation + 1, num_relation + 2
    max_pred, max_len = max_pred_num, max_len
    pure_relation_vocab = list(range(0, num_relation))
    relation_vocab = list(range(0, num_relation + 3))
    mask_prob = mask_prob

    def mask_row(row):
        label, relation_a, relation_b, rel_a_len, rel_b_len = row['label'], row['p1'], row['p2'], row['p_len1'], \
                                                              row['p_len2']
        relation_pair = np.array([CLS] + relation_a[:rel_a_len] + [SEP] + relation_b[:rel_b_len] + [SEP]).astype(np.int16)
        position_pair = np.array([0] + list(range(1, (rel_a_len + 1))) + [rel_a_len + 1] + list(range(1, (rel_b_len + 1))) + [rel_b_len + 1]).astype(np.int16)
        input_mask = np.full((len(relation_pair)), fill_value=1, dtype=np.int8)

        masked_rels, masked_pos = [], []
        num_pred = min(max_pred, max(1, int(round(len(relation_pair) * mask_prob))))
        cand_pos = [i for i, relation in enumerate(relation_pair) if relation != CLS and relation != SEP]
        shuffle(cand_pos)

        def get_random_relation(relation_vocab, relation):
            rand_rel = random.choice(relation_vocab)
            while rand_rel == relation:
                rand_rel = random.choice(relation_vocab)
            return rand_rel

        for pos in cand_pos[:num_pred]:
            masked_rels.append(relation_pair[pos])
            masked_pos.append(pos)
            if rand() < 0.8:
                relation_pair[pos] = MASK
            elif rand() < 0.5:
                relation_pair[pos] = get_random_relation(pure_relation_vocab, relation_pair[pos])

        masked_rels = np.array(masked_rels).astype(np.int16)
        masked_pos = np.array(masked_pos).astype(np.int16)
        masked_weights = np.full((num_pred), fill_value=1, dtype=np.int8)

        n_pad = max_len - len(relation_pair)
        relation_pair = np.pad(relation_pair, (0, n_pad), 'constant')
        input_mask = np.pad(input_mask, (0, n_pad), 'constant')
        position_pair = np.pad(position_pair, (0, n_pad), 'constant')

        n_pad_mask = max_pred - num_pred
        if n_pad_mask > 0:
            masked_rels = np.pad(masked_rels, (0, n_pad_mask), 'constant')
            masked_pos = np.pad(masked_pos, (0, n_pad_mask), 'constant')
            masked_weights = np.pad(masked_weights, (0, n_pad_mask), 'constant')
        return relation_pair.astype(np.int16), position_pair.astype(np.int16), \
               input_mask.astype(np.int8), masked_rels.astype(np.int16), \
               masked_pos.astype(np.int16), masked_weights.astype(np.int8)
    data[['r_pair', 'pos_pair', 'input_mask', 'mask_rel', 'mask_pos', 'mask_mask']] = data.parallel_apply(mask_row, axis=1, result_type='expand')
    drop_col_names = [col for col in data.columns if col not in {'label', 'r_pair', 'pos_pair', 'input_mask',
                                                                 'mask_rel', 'mask_pos', 'mask_mask'}]
    data = data.drop(columns=drop_col_names)
    return data, relation_vocab


if __name__ == '__main__':
    pair_data_path = '../SeqData/RelPathPair_WN18RR/'
    seed = 2019
    epochs = 200
    walk_len = 8
    samp_type = 'weighted'
    # samp_type = 'uniform'
    num_of_pairs = 5000000
    set_seeds(seed)
    relation_num = 11

    rel_path_pair_data_file_name = 'Relation_Walk_Epoch_' + str(epochs) + "_walklen_" + str(walk_len) + '_seed_' + str(
        seed) + '_pairnum_' + str(num_of_pairs) + '_samp_type_' + samp_type
    # save_to_json(data=path_pair_data, file_name=pair_data_path + rel_path_pair_data_file_name + '.json')
    df = load_json_as_data_frame(pair_data_path + rel_path_pair_data_file_name + '.json')

    print(df['label'].value_counts(normalize = True))

    max_len = 2*(walk_len - 1) + 3
    start = time()
    data, _ = mask_relation_pair_data(data=df, num_relation=relation_num, max_len=max_len)
    for col in data.columns:
        print(col)
    print(time() - start)

    rel_path_pair_data_file_masked_name = 'Masked_Relation_Walk_Epoch_' + str(epochs) + "_walklen_" + str(walk_len) + '_seed_' + str(
        seed) + '_pairnum_' + str(num_of_pairs) + '_samp_type_' + samp_type + '.json'
    save_to_json(data=data, file_name=pair_data_path + rel_path_pair_data_file_masked_name)

