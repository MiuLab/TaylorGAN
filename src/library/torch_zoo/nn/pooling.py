import torch
from torch.nn import Module


class GlobalAvgPool1D(Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=self.dim)
