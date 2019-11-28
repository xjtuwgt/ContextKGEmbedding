import torch
import torch.nn as nn
from torch.optim import Adam
from KGBERTPretrain.kg_language_model import BERT, BERTLM
from ModelTrainer.optim_schedule import ScheduledOptim
from KGEmbedUtils.ioutils import save_model

class KGBERTTrainer():
    def __init__(self, bert: BERT, rel_vocab_size: int, train_dataloader, batch_num,
                 test_dataloader=None, num_class=2,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, log_freq: int = 10, num_gpu=0):
        """
        :param bert:
        :param vocab_size:
        :param train_dataloader:
        :param test_dataloader:
        :param lr:
        :param betas:
        :param weight_decay:
        :param warmup_steps:
        :param with_cuda:
        :param cuda_devices:
        :param log_freq:
        """
        # cuda_condition = torch.cuda.is_available() and with_cuda
        # self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        device_ids = None
        if num_gpu >= 1:
            print('Using GPU!')
            if num_gpu > 1:
                device_ids = range(num_gpu)
            aa = []
            for i in range(num_gpu - 1, -1, -1):
                device = torch.device('cuda: % d' % i)
                aa.append(torch.rand(1).to(device))  # a place holder
        else:
            device = 'cpu'
            print('Using CPU!')

        self.device = device
        self.bert = bert
        self.batch_num = batch_num
        self.model = BERTLM(bert=bert, rel_vocab_size=rel_vocab_size, num_class=num_class).to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=device_ids)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion_pair = nn.NLLLoss()
        self.criterion_mlm = nn.NLLLoss(reduction='none')

        self.log_freq = log_freq
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        acc = self.iteration(epoch=epoch, data_iterator=self.train_data)
        return acc

    def iteration(self, epoch, data_iterator, train=True):
        """
        :param epoch:
        :param data_iterator:
        :param train:
        :return:
        """
        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        batch_idx = 0
        while batch_idx < self.batch_num:
            batch = next(data_iterator)
            mode = batch['mode']
            batch = {key: value.to(self.device) for key, value in batch.items() if key != 'mode'}
            rel_pair_out_put, mask_lm_output = self.model.forward(batch['rel_seq'], batch['pos_seq'], batch['mask_seq'])
            pair_loss = self.criterion_pair(rel_pair_out_put, batch['label'])

            mask_pos_lm = batch['m_pos_seq']
            mask_pos_lm = mask_pos_lm[:,:,None].expand(-1,-1,mask_lm_output.shape[-1])
            lm_preds = torch.gather(mask_lm_output, 1, mask_pos_lm)
            lm_labels = batch['m_rel_seq']
            mask_mask = batch['m_mask_seq']
            lm_loss = self.criterion_mlm(lm_preds.transpose(1,2), lm_labels)
            mask_loss = (lm_loss * mask_mask).mean()

            loss = pair_loss + mask_loss

            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            correct = rel_pair_out_put.argmax(dim=-1).eq(batch["label"]).sum().item()
            # print(correct, batch['label'].sum())
            avg_loss += loss.item()
            total_correct += correct
            total_element += batch["label"].nelement()

            post_fix = {
                "epoch": epoch,
                "iter": batch_idx,
                "avg_loss": avg_loss / (batch_idx + 1),
                "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }

            if batch_idx % self.log_freq == 0:
                print(str(post_fix))
            batch_idx = batch_idx + 1
        print("EP%d_%s, avg_loss=" % (epoch, 'train'), avg_loss / (self.batch_num), "total_acc=",
              total_correct * 100.0 / total_element)
        return '{:.2f}'.format(total_correct * 100.0 / total_element)

    def save(self, epoch, file_path='../SavedModel/bert', acc=None):
        if acc is None:
            out_put_path = file_path + '.ep%d' % epoch
        else:
            out_put_path = file_path + '_acc_' + str(acc) + '.ep%d' % epoch
        # torch.save(self.bert.cpu().state_dict(), out_put_path)
        save_model(self.bert.cpu(), out_put_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, out_put_path)
        return out_put_path
