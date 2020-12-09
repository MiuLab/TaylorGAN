import tensorflow as tf

from library.my_argparse.actions import IdKwargs
from library.tf_keras_zoo.optimizers import GradientClipping, LookAhead, RAdamOptimizer, WeightDecay
from library.utils import allow_abbrev_kwargs, format_object, logging_block

from .base import get_help_of_id_kwargs


optimizer_cls_table = dict(
    sgd=tf.train.GradientDescentOptimizer,
    rmsprop=tf.train.RMSPropOptimizer,
    adam=tf.train.AdamOptimizer,
    radam=RAdamOptimizer,
)


def create(args, module_name):
    optimizer_id, kwargs = getattr(args, f'{module_name[0]}_optimizer')
    return _create(optimizer_id, **kwargs)


@allow_abbrev_kwargs
def _create(
        optimizer_id: str,
        learning_rate: float,
        clip_value: float = None,
        clip_norm: float = None,
        clip_global_norm: float = None,
        weight_decay_rate: float = None,
        use_lookahead: bool = False,
        **kwargs,
    ):
    optimizer = optimizer_cls_table[optimizer_id](learning_rate, **kwargs)
    with logging_block(
        f"Optimizer: {format_object(optimizer, learning_rate=learning_rate, **kwargs)}",
    ):
        if clip_value:
            print(f"clip_value: {clip_value}")
            optimizer = GradientClipping(optimizer, clip_value, clip_by='value')
        elif clip_norm:
            print(f"clip_norm: {clip_norm}")
            optimizer = GradientClipping(optimizer, clip_norm, clip_by='norm')
        elif clip_global_norm:
            print(f"clip_global_norm: {clip_global_norm}")
            optimizer = GradientClipping(optimizer, clip_global_norm, clip_by='global_norm')

        if weight_decay_rate:
            print(f"weight_decay_rate: {weight_decay_rate}")
            optimizer = WeightDecay(optimizer, decay_rate=weight_decay_rate * learning_rate)

        if use_lookahead:
            print("use_lookahead: True")
            optimizer = LookAhead(optimizer)

    return optimizer


def add_argument_to(holder, module_name):
    holder.add_argument(
        f'--{module_name[0]}-optimizer',
        action=IdKwargs,
        id_choices=optimizer_cls_table.keys(),
        default=IdKwargs.IdKwargsPair(
            'adam', learning_rate=1e-4, beta1=0.5, clip_global_norm=10,
        ),
        split_token=',',
        metavar='OPTIMIZER_ID',
        help=f"{module_name}'s optimizer settings.\n"
             "Shared options:\n"
             "Optimizer(clip_value: float, clip_norm: float, "
             "clip_global_norm: float, weight_decay_rate: float, use_lookahead: bool)\n"
             f"{get_help_of_id_kwargs(optimizer_cls_table)}\n",
    )
