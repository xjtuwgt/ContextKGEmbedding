import torch
import numpy as np
from torch import Tensor
from torch import autograd
from KGBERTPretrain.transformer import make_model
def data_gen(V, batch, nbatches):
    "Generate random data for a src copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 3)))
        data[:, 0] = 1
        src = autograd.Variable(data, requires_grad=False)
        yield Batch(src, 0)

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src: Tensor, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)

if __name__ == '__main__':
    torch.manual_seed(2019)
    np.random.seed(2019)
    batch = data_gen(100, 2, 1)
    encoder = make_model(vocab=100, num_layers=80, num_features=18, num_head=2, dropout=0.05)
    for i, batch in enumerate(batch):
        # print(i, batch.src, batch.src_mask)
        x = encoder.forward(batch.src, batch.src_mask)
        print(x)