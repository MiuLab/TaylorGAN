import abc

from library.utils import FormatableMixin

from ..collections import LossCollection


class Regularizer(abc.ABC, FormatableMixin):

    def __init__(self, coeff: float):
        self.coeff = coeff

    def __call__(self, **kwargs) -> LossCollection:
        loss = self.compute_loss(**kwargs)
        return LossCollection(self.coeff * loss, **{self.loss_name: loss})

    @property
    @abc.abstractmethod
    def loss_name(self) -> str:
        pass

    @abc.abstractmethod
    def compute_loss(self):
        pass
