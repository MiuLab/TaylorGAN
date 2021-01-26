from functools import lru_cache

import torch

from core.models import ModuleInterface
from .base import Regularizer, LossCollection


class VariableRegularizer(Regularizer):

    def __call__(self, generator=None, discriminator=None, **kwargs) -> LossCollection:
        if generator and discriminator:
            raise TypeError
        return self.compute_loss(module=generator or discriminator)


class EmbeddingRegularizer(VariableRegularizer):

    loss_name = 'embedding'

    def __init__(self, max_norm: float = 0.):
        self.max_norm = max_norm

    def compute_loss(self, module: ModuleInterface):
        embedding_L2_loss = torch.square(module.embedding_matrix).sum(dim=1)  # shape (V, )
        if self.max_norm:
            embedding_L2_loss = (embedding_L2_loss - self.max_norm ** 2).clamp(min=0.)
        return embedding_L2_loss.mean() / 2  # shape ()


class SpectralRegularizer(VariableRegularizer):

    loss_name = 'spectral'

    def compute_loss(self, module: ModuleInterface):
        loss = 0
        for module in module.modules():
            weight = getattr(module, 'weight', None)
            if weight is None:
                continue
            sn, u, new_u = self._get_spectral_norm(weight)
            loss += (sn ** 2) / 2
            u.copy_(new_u)

        return loss

    def _get_spectral_norm(self, weight: torch.nn.Parameter):
        u = get_u(weight)  # shape (U)
        if weight.ndim > 2:
            weight_matrix = weight.view(weight.shape[0], -1)
        else:
            weight_matrix = weight  # shape (U, V)

        v = torch.nn.functional.normalize(torch.mv(weight_matrix.t(), u), dim=0).detach()
        Wv = torch.mv(weight_matrix, v)  # shape (U)
        new_u = torch.nn.functional.normalize(Wv, dim=0).detach()  # shape (U)
        spectral_norm = torch.tensordot(new_u, Wv, dims=1)
        return spectral_norm, u, new_u


@lru_cache(None)
def get_u(kernel):
    return kernel.new_empty(kernel.shape[0]).normal_().detach()
