import torch as th
from torch.nn import Module


class GlobalAvgPool1D(Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return th.mean(x, dim=self.dim)
