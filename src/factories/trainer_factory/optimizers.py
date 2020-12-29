from functools import wraps

import torch

from core.train.optimizer import OptimizerWrapper
from flexparse.types import FactoryMethod
from library.utils import match_abbrev

from ..utils import create_factory_action


def create_action_of(module_name: str):
    shared_options = ["clip_value", "clip_norm", "use_lookahead"]
    return create_factory_action(
        f'--{module_name[0]}-optimizer',
        registry={
            'sgd': wrap_optimizer(torch.optim.SGD),
            'rmsprop': wrap_optimizer(torch.optim.RMSprop),
            'adam': wrap_optimizer(torch.optim.Adam),
        },
        default='adam(1e-4, clip_norm=10)',
        help_prefix=(
            f"{module_name}'s optimizer.\n\n"
            f"shared options: {FactoryMethod.COMMA.join(shared_options)}\n"
        ),
    )


def wrap_optimizer(optimizer_func):

    @match_abbrev
    @wraps(optimizer_func)
    def wrapper(*args, clip_value: float = None, clip_norm: float = None, **kwargs):

        def partial(params):
            optimizer = optimizer_func(params, *args, **kwargs)
            return OptimizerWrapper(optimizer, clip_value=clip_value, clip_norm=clip_norm)

        return partial

    return wrapper
