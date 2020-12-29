from itertools import chain

import torch

from library.utils import format_object, ObjectWrapper


class OptimizerWrapper(ObjectWrapper):

    '''Just for late printing'''

    def __init__(self, optimizer, clip_value, clip_norm):
        super().__init__(optimizer)
        self.clip_value = clip_value
        self.clip_norm = clip_norm

    def step(self, closure=None):
        torch.nn.utils.clip_grad_norm_(self.params, self.clip_norm)
        self._body.step(closure)

    @property
    def params(self):
        return chain.from_iterable(
            group['params']
            for group in self._body.param_groups
        )

    def __str__(self):
        kwargs = {
            key: val
            for key, val in self._body.param_groups[0].items()
            if key != 'params'
        }
        if self.clip_value:
            kwargs['clip_value'] = self.clip_value
        if self.clip_norm:
            kwargs['clip_norm'] = self.clip_norm

        return format_object(self._body, **kwargs)
