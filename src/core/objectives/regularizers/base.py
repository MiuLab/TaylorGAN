import abc

import torch

from library.utils import ObjectWrapper, wraps_with_new_signature, format_object

from ..collections import LossCollection


class Regularizer(abc.ABC):

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

    @classmethod
    def as_constructor(cls, regularizer_cls):

        @wraps_with_new_signature(regularizer_cls)
        def wrapper(coeff, *args, **kwargs):
            return cls(
                regularizer=regularizer_cls(*args, **kwargs),
                coeff=coeff,
            )

        return wrapper

    def __str__(self):
        params = {'coeff': self.coeff, **self.regularizer.__dict__}
        return format_object(self.regularizer, **params)
