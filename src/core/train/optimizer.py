from itertools import chain

import torch

from library.utils import format_object, ObjectWrapper, wraps_with_new_signature


class OptimizerWrapper(ObjectWrapper):

    def __init__(self, optimizer, clip_norm: float = 0):
        super().__init__(optimizer)
        self.optimizer = optimizer
        self.clip_norm = clip_norm

    def step(self, closure=None):
        torch.nn.utils.clip_grad_norm_(self.params, self.clip_norm)
        self.optimizer.step(closure)

    @property
    def params(self):
        return chain.from_iterable(
            group['params']
            for group in self.optimizer.param_groups
        )

    def __str__(self):
        kwargs = {
            key: val
            for key, val in self.optimizer.param_groups[0].items()
            if key != 'params'
        }
        if self.clip_norm:
            kwargs['clip_norm'] = self.clip_norm

        return format_object(self.optimizer, **kwargs)

    @classmethod
    def as_constructor(cls, optimizer_cls):

        @wraps_with_new_signature(optimizer_cls)
        def wrapper(*args, clip_norm=0, **kwargs):
            return cls(
                optimizer=optimizer_cls(*args, **kwargs),
                clip_norm=clip_norm,
            )

        return wrapper
