import inspect
from functools import wraps

import torch
from flexparse import LookUpPartial

from core.train.optimizer import OptimizerWrapper

from ..utils import create_factory_action


def create_action_of(module_name: str):
    return create_factory_action(
        f'--{module_name[0]}-optimizer',
        type=LookUpPartial(
            {
                'sgd': wrap_optimizer(torch.optim.SGD),
                'rmsprop': wrap_optimizer(torch.optim.RMSprop),
                'adam': wrap_optimizer(torch.optim.Adam),
            },
            target_signature=['params'],
        ),
        default='adam(lr=1e-4, betas=(0.5, 0.999), clip_norm=10)',
        help_prefix=f"{module_name}'s optimizer.\n",
    )


def wrap_optimizer(optimizer_func):

    @wraps(optimizer_func)
    def wrapper(*args, clip_norm=0, **kwargs):
        optimizer = optimizer_func(*args, **kwargs)
        return OptimizerWrapper(optimizer, clip_norm=clip_norm)

    old_sig = inspect.signature(optimizer_func)
    add_params = [
        p
        for p in inspect.signature(wrapper, follow_wrapped=False).parameters.values()
        if p.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    ]
    new_sig = old_sig.replace(parameters=[*old_sig.parameters.values(), *add_params])
    wrapper.__signature__ = new_sig
    return wrapper
