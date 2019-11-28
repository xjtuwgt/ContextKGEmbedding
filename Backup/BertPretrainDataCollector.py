import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from Backup.RelPathPairCollector import relation_path_pair_generater
from Backup.PretrainedDataForKG import mask_relation_pair_data
from time import time
from KGEmbedUtils.ioutils import save_to_json, load_json_as_data_frame
from KGEmbedUtils.utils import set_seeds

def relation_pair_generator(random_walk_path, random_walk_data_name: str, num_of_pairs:int, samp_type, rel_pair_path:str):
    reader = relation_path_pair_generater(json_relation_pair_file_name=random_walk_path +
                                                                       random_walk_data_name + '.json', samp_type=samp_type)
    start = time()
    path_pair_data = reader.relation_pair_generation_fast(num_pairs=num_of_pairs)
    print('runtime = {}'.format(time() - start))
    print(path_pair_data['label'].value_counts(normalize=True, sort=True))
    rel_path_pair_data_file_name = 'Rel_Pair_' + random_walk_data_name +'_pairnum_' + str(num_of_pairs) + '_samp_type_' + samp_type
    save_to_json(data=path_pair_data, file_name=rel_pair_path + rel_path_pair_data_file_name + '.json')
    print(rel_pair_path + rel_path_pair_data_file_name + '.json')
    return rel_path_pair_data_file_name

def relation_path_mask(rel_pair_path:str, rel_path_pair_data_file_name:str, walk_len:int, relation_num:int):
    df = load_json_as_data_frame(rel_pair_path + rel_path_pair_data_file_name + '.json')
    print(df['label'].value_counts(normalize=True))

    max_len = 2 * (walk_len - 1) + 3
    start = time()
    data, _ = mask_relation_pair_data(data=df, num_relation=relation_num, max_len=max_len)
    for col in data.columns:
        print(col)
    print(time() - start)
    rel_path_pair_data_file_masked_name = 'Masked_'+ rel_path_pair_data_file_name +'.json'
    save_to_json(data=data, file_name=rel_pair_path + rel_path_pair_data_file_masked_name)
    print(rel_pair_path + rel_path_pair_data_file_masked_name)


if __name__ == '__main__':
    seed = 2019
    set_seeds(seed)
    rand_walk_path = '../SeqData/RandWalk_WN18RR/'
    rel_pair_data_path = '../SeqData/RelPathPair_WN18RR/'
    rand_walk_data_file_name = 'Relation_Walk_Epoch_1_walklen_8_seed_2019'
    samp_type = 'weighted'
    num_pairs = 5000000
    walk_len = 8
    relation_num = 11
    rel_pair_data_file_name = relation_pair_generator(random_walk_path=rand_walk_path,
                                                      random_walk_data_name=rand_walk_data_file_name,
                                                      rel_pair_path=rel_pair_data_path, samp_type=samp_type, num_of_pairs=num_pairs)
    relation_path_mask(rel_pair_path=rel_pair_data_path, rel_path_pair_data_file_name=rel_pair_data_file_name, walk_len=walk_len, relation_num=11)