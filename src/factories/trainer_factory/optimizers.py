import tensorflow as tf

from core.train.optimizer import OptimizerWrapper
from flexparse.types import FactoryMethod
from library.tf_keras_zoo.optimizers import GradientClipping, LookAhead, RAdamOptimizer, WeightDecay
from library.utils import match_abbrev

from ..utils import create_factory_action


def create_action_of(module_name: str):
    shared_options = [
        "clip_value", "clip_norm", "clip_global_norm", "weight_decay_rate", "use_lookahead",
    ]
    return create_factory_action(
        f'--{module_name[0]}-optimizer',
        registry={
            'sgd': sgd,
            'rmsprop': rmsprop,
            'adam': adam,
            'radam': radam,
        },
        default='adam(1e-4,beta1=0.5,clip_global_norm=10)',
        help_prefix=(
            f"{module_name}'s optimizer.\n\n"
            f"shared options: {FactoryMethod.COMMA.join(shared_options)}\n"
        ),
    )


def sgd(learning_rate: float, **kwargs):
    return _wrap_optimizer(
        tf.train.GradientDescentOptimizer,
        learning_rate,
        **kwargs,
    )


def rmsprop(learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10, centered=False, **kwargs):
    return _wrap_optimizer(
        tf.train.RMSPropOptimizer,
        learning_rate,
        decay=decay, momentum=momentum, epsilon=epsilon, centered=centered,
        **kwargs,
    )


def adam(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, **kwargs):
    return _wrap_optimizer(
        tf.train.AdamOptimizer,
        learning_rate,
        beta1=beta1, beta2=beta2, epsilon=epsilon,
        **kwargs,
    )


def radam(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, **kwargs):
    return _wrap_optimizer(
        RAdamOptimizer,
        learning_rate,
        beta1=beta1, beta2=beta2, epsilon=epsilon,
        **kwargs,
    )


@match_abbrev
def _wrap_optimizer(
        optimizer_cls,
        learning_rate,
        clip_value: float = None,
        clip_norm: float = None,
        clip_global_norm: float = None,
        weight_decay_rate: float = None,
        use_lookahead: bool = False,
        **optimizer_params,
    ):
    optimizer = optimizer_cls(learning_rate, **optimizer_params)
    wrapper_info = []
    if clip_value:
        optimizer = GradientClipping(optimizer, clip_value, clip_by='value')
        wrapper_info.append(f"clip_value: {clip_value}")
    elif clip_norm:
        optimizer = GradientClipping(optimizer, clip_norm, clip_by='norm')
        wrapper_info.append(f"clip_norm: {clip_norm}")
    elif clip_global_norm:
        optimizer = GradientClipping(optimizer, clip_global_norm, clip_by='global_norm')
        wrapper_info.append(f"clip_global_norm: {clip_global_norm}")

    if weight_decay_rate:
        optimizer = WeightDecay(optimizer, decay_rate=weight_decay_rate * learning_rate)
        wrapper_info.append(f"weight_decay_rate: {weight_decay_rate}")

    if use_lookahead:
        optimizer = LookAhead(optimizer)
        wrapper_info.append("use_lookahead: True")

    return OptimizerWrapper(
        optimizer,
        wrapper_info=wrapper_info,
        learning_rate=learning_rate,  # attr names are different for tf Optimizer subclasses...
        **optimizer_params,
    )
