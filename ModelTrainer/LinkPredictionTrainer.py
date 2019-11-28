import torch
import torch.nn as nn
from torch.optim import Adam
from KGBERTPretrain.kg_language_model import BERT
from CoRelKGModel.triple2pathDataLoader import BidirectionalOneShotIterator
from ModelTrainer.optim_schedule import ScheduledOptim
from CoRelKGModel.kgePrediction import ContextKGEModel
from KGEmbedUtils.ioutils import save_model
from time import time

class LinkPredictionTrainer():
    def __init__(self, bert: BERT, num_relations: int, num_entity: int, hid_dim: int,
                 train_dataloader: BidirectionalOneShotIterator,
                gamma=0.1, lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, pool_type='attn',
                with_cuda: bool = True, regularization=0.001, warmup_steps=10000, log_freq: int = 100, num_gpu=0):
        device_ids = None
        if num_gpu >= 1:
            print('Using GPU!')
            if num_gpu > 1:
                device_ids = range(num_gpu)
            aa = []
            for i in range(num_gpu - 1, -1, -1):
                device = torch.device('cuda: % d' % i)
                aa.append(torch.rand(1).to(device))  # a place holder
            self.use_cuda = True
        else:
            self.use_cuda = False
            device = 'cpu'
            print('Using CPU!')
        self.device = device
        self.log_freq = log_freq
        self.regularization=regularization
        if self.use_cuda:
            bert = bert.to(self.device)
        self.model = ContextKGEModel(num_relation=num_relations, num_entity=num_entity, hidden_dim=hid_dim,
                                     bert_path_encoder=bert, gamma=gamma, pool_type=pool_type).to(self.device)

        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, bert.hidden, n_warmup_steps=warmup_steps)

        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=device_ids)

        self.train_data = train_dataloader

    def train(self, num_steps, save_path: str):
        logs = []
        start = time()
        print('Training start...')
        for step in range(1, num_steps):
            loss = self.iteration()
            torch.cuda.empty_cache()
            logs.append(loss.detach())
            if step % self.log_freq == 0:
                avg_loss = sum(logs)/len(logs)
                print('[Train]({}/{}) average {}'.format(step, num_steps, avg_loss))
                print('[Train] {} steps take {:.3f} seconds'.format(self.log_freq,
                                                                     time() - start))
                start = time()
            if step > 0 and step % 5000 == 0:
                avg_loss = sum(logs) / len(logs)
                self.save(step, loss=avg_loss, file_path=save_path)

        avg_loss = sum(logs)/len(logs)
        self.save(num_steps, loss=avg_loss, file_path=save_path)
        return avg_loss

    def iteration(self):
        loss = self.model.train_step(self.model, self.optim, self.optim_schedule, self.train_data, device=self.device, regularization=self.regularization)
        return loss

    def save(self, batch_num, loss, file_path):
        # torch.save(self.bert.cpu(), out_put_path)
        out_put_path = file_path + '_loss_' + '{:.3f}'.format(loss) + '.bt%d' % batch_num
        save_model(model=self.model.cpu(), model_path=out_put_path)
        self.model.to(self.device)
        print("Model Saved on:", out_put_path)
        return out_put_path