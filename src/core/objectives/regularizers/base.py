import abc

import torch

from library.utils import FormatableMixin, ObjectWrapper

from ..collections import LossCollection


class Regularizer(abc.ABC, FormatableMixin):

    @abc.abstractmethod
    def __call__(self, **kwargs) -> LossCollection:
        pass

    @property
    @abc.abstractmethod
    def loss_name(self) -> str:
        pass


class LossScaler(ObjectWrapper):

    def __init__(self, regularizer: Regularizer, coeff: float):
        super().__init__(regularizer)
        self.regularizer = regularizer
        self.coeff = coeff

    def __call__(self, **kwargs):
        loss = self.regularizer(**kwargs)
        if isinstance(loss, torch.Tensor):
            observables = {self.regularizer.loss_name: loss}
        else:
            loss, observables = loss

        return LossCollection(self.coeff * loss, **observables)
