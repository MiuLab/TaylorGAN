import torch
from flexparse import LookUpCall

from core.train.optimizer import OptimizerWrapper
from library.utils import ArgumentBinder, wraps_with_new_signature

from ..utils import create_factory_action


def create_action_of(module_name: str):
    return create_factory_action(
        f'--{module_name[0]}-optimizer',
        type=LookUpCall(
            {
                key: ArgumentBinder(wrap_optimizer(optim_cls), preserved=['params'])
                for key, optim_cls in [
                    ('sgd', torch.optim.SGD),
                    ('rmsprop', torch.optim.RMSprop),
                    ('adam', torch.optim.Adam),
                ]
            },
        ),
        default='adam(lr=1e-4, betas=(0.5, 0.999), clip_norm=10)',
        help_prefix=f"{module_name}'s optimizer.\n",
    )


def wrap_optimizer(optimizer_func):

    @wraps_with_new_signature(optimizer_func)
    def wrapper(*args, clip_norm=0, **kwargs):
        return OptimizerWrapper(
            optimizer=optimizer_func(*args, **kwargs),
            clip_norm=clip_norm,
        )

    return wrapper
