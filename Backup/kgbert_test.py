from KGBERTPretrain.bert import BERT
import torch

if __name__ == '__main__':
    model = BERT(rel_vocab_size=14)

    rel_seq = torch.randint(0, 8, size=(10, 17))
    pos_seq = torch.randint(0, 8, size=(10, 17))
    mask_seq = torch.randint(0,2, size=(10,17))
    print(mask_seq)

    # print(rel_seq.shape, pos_seq.shape, mask_seq.shape)
    #
    print(model(rel_seq, pos_seq, mask_seq))

