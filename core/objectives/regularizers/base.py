import abc

from library.utils import format_object

from ..collections import LossCollection


class Regularizer(abc.ABC):

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

    def __str__(self):
        return format_object(self, get_attrs=True)
