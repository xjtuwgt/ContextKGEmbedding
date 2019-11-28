import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import argparse
from ModelDataLoader.bertDataLoader import construct_train_iterator
from KGEmbedUtils.kgdataUtils import dataloader
from KGEmbedUtils.ioutils import load_json_as_data_frame
from KGBERTPretrain.bert import BERT
from ModelTrainer.bertTrainer import KGBERTTrainer
from KGEmbedUtils.utils import set_seeds
from pandas import DataFrame


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--train_dataset", required=False, type=str, default='RandWalk_walklen_5_epoch_4000_seed_2019', help="train dataset for train bert")
    parser.add_argument("-v", "--num_rel", required=False, type=int, default=11, help="number of relations")
    parser.add_argument("--max_len", required=False, type=int, default=8, help="maximum relation path length")
    parser.add_argument("-o", "--output_path", required=False, type=str, default='../SavedModel/WN18RR/', help="model save path")
    parser.add_argument("-i", "--input_path", required=False, type=str, default='../SeqData/RelPathPair_WN18RR', help="input data path")
    parser.add_argument("--data_name", required=False, type=str, default='WN18RR', help="input data name")

    parser.add_argument("-hs", "--hidden", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=6, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=4, help="number of attention heads")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("--samp_size", type=int, default=10, help="number of batch_size")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.00001, help="weight_decay of adam")

    parser.add_argument("--num_class", type=int, default=2, help="number of class")
    parser.add_argument("-e", "--epochs", type=int, default=30, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=8, help="dataloader worker size")
    parser.add_argument("-s", "--seq_len", type=int, default=12, help="maximum sequence len")

    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")
    parser.add_argument("--with_cuda", type=bool, default=True)
    parser.add_argument("--num_gpu", type=int, default=0, help="number of gpu")

    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")
    parser.add_argument("--rand_seed", type=int, default=2019, help="random seed")

    args = parser.parse_args()
    seed = args.rand_seed
    set_seeds(seed)

    args = parser.parse_args()
    for key, value in vars(args).items():
        print('Parameter setting {} = {}'.format(key, value))

    print("Loading Train Dataset", args.train_dataset)
    data = dataloader(args.data_name)
    num_nodes = data.num_entities
    num_relations = data.num_relation
    input_path = args.input_path
    path_pair_name = args.train_dataset + '_path_pair.json'
    all_true_name = args.train_dataset + '_uniq_path.json'

    path_pir_df: DataFrame = load_json_as_data_frame(os.path.join(input_path, path_pair_name))
    true_path_df = load_json_as_data_frame(os.path.join(input_path, all_true_name))

    print("Creating Dataloader")
    train_data_loader = construct_train_iterator(num_entity=num_nodes, num_relations=num_relations, ent2path=path_pir_df, true_path_df=true_path_df,
                                                  sample_size=args.samp_size, batch_size=args.batch_size)

    print("Building BERT model")
    bert = BERT(vocab_size=num_nodes + num_relations, max_rel_len=args.seq_len, hidden=args.hidden, head_num=args.attn_heads, n_layers=args.layers)

    print("Creating BERT Trainer")
    batch_num = num_nodes // args.batch_size if num_nodes % args.batch_size == 0 else (num_nodes // args.batch_size + 1)
    trainer = KGBERTTrainer(bert, rel_vocab_size=num_relations, train_dataloader=train_data_loader, test_dataloader=None, num_class=args.num_class,
                          lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                          with_cuda=args.with_cuda, log_freq=args.log_freq, num_gpu=args.num_gpu, batch_num=batch_num)

    print("Training Start")
    save_path = args.output_path + args.train_dataset + '_hid_' + str(args.hidden) + '_layer_' + str(args.layers) + \
                '_heads_' + str(args.attn_heads) + '_bs_' + str(args.batch_size) + '_lr_' + str(args.lr) + \
                '_l2_' + str(args.adam_weight_decay) + '_class_num_' + str(args.num_class) + '_bert'
    print('Path saved = {}'.format(save_path))
    for epoch in range(args.epochs):
        acc = trainer.train(epoch)
        trainer.save(epoch, save_path, acc=acc)

if __name__ == '__main__':
    train()
