import tensorflow as tf

from core.objectives.GAN import (
    GANLossTuple,
    BCE,
    GumbelSoftmaxGAN,
    ReinforceGAN,
    SoftmaxGAN,
    StraightThroughSoftmaxGAN,
    GumbelStraightThroughSoftmaxGAN,
    TaylorGAN,
)
from factories.base import SimpleFactory, get_help_of_id_kwargs
from library.my_argparse.actions import IdKwargs


class LossFactory(SimpleFactory):

    def create(self, args):
        return self.table[args.loss]

    def register(self, key: str, loss):
        self.table[key] = loss

    def add_argument_to(self, holder):
        holder.add_argument(
            '--loss',
            choices=self.table.keys(),
            default='RKL',
            help='Loss function type.',
        )


class EstimatorFactory(SimpleFactory):

    def create(self, args, **kwargs):
        est_id, est_kwargs = args.estimator
        return self.table[est_id](**est_kwargs, **kwargs)

    def add_argument_to(self, holder):
        holder.add_argument(
            '--estimator',
            action=IdKwargs,
            id_choices=self.table.keys(),
            default=IdKwargs.IdKwargsPair('taylor'),
            split_token=',',
            metavar='ESTIMATOR_ID',
            help=f"The estimator for discrete sampling.\n{get_help_of_id_kwargs(self.table)}\n",
        )


loss_factory = LossFactory({
    'alt': GANLossTuple(lambda fake_score: BCE(fake_score, labels=1.)),  # RKL - 2JS
    'JS': GANLossTuple(lambda fake_score: -BCE(fake_score, labels=0.)),  # 2JS
    'KL': GANLossTuple(lambda fake_score: -tf.exp(fake_score)),  # -sig / (1 - sig)
    'RKL': GANLossTuple(lambda fake_score: -fake_score),  # log((1 - sig) / sig)
})
estimator_factory = EstimatorFactory({
    'taylor': TaylorGAN,
    'reinforce': ReinforceGAN,
    'straight-through': StraightThroughSoftmaxGAN,
    'gumbel': GumbelSoftmaxGAN,
    'st-gumbel': GumbelStraightThroughSoftmaxGAN,
    'softmax': SoftmaxGAN,
})
