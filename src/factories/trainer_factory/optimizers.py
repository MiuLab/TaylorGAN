import torch
from flexparse import LookUpCall

from core.train.optimizer import OptimizerWrapper
from library.utils import ArgumentBinder

from ..utils import create_factory_action


def create_action_of(module_name: str):
    return create_factory_action(
        f'--{module_name[0]}-optimizer',
        type=LookUpCall(
            {
                key: ArgumentBinder(
                    OptimizerWrapper.as_constructor(optim_cls),
                    preserved=['params'],
                )
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
