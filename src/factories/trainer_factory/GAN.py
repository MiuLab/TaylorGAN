import torch

from core.models import Discriminator
from core.objectives.GAN import (
    BCE,
    GANObjective,
    GANLossTuple,
    ReinforceEstimator,
    StraightThroughEstimator,
    TaylorEstimator,
    GumbelSoftmaxEstimator,
)
from core.train import DiscriminatorUpdater, GANTrainer
from factories.modules import discriminator_factory
from flexparse import create_action, LookUp, LookUpCall, IntRange
from library.utils import cached_property

from ..utils import create_factory_action
from .trainer_factory import TrainerCreator, create_optimizer_action_of


class GANCreator(TrainerCreator):

    def create_trainer(self, generator_updater) -> GANTrainer:
        loss_tuple, _, d_steps = self.args[GAN_ARGS]
        return GANTrainer(
            generator_updater=generator_updater,
            discriminator_updater=self.create_discriminator_updater(
                self._discriminator,
                discriminator_loss=loss_tuple.discriminator_loss,
            ),
            d_steps=d_steps,
        )

    def create_discriminator_updater(self, discriminator, discriminator_loss):
        return DiscriminatorUpdater(
            discriminator,
            optimizer=self.args[D_OPTIMIZER_ARG](discriminator.trainable_variables),
            losses=[
                discriminator_loss,
                *self.args[discriminator_factory.REGULARIZER_ARG],
            ],
        )

    @cached_property
    def objective(self):
        loss_tuple, estimator = self.args[GAN_ARGS[:2]]
        return GANObjective(
            discriminator=self._discriminator,
            generator_loss=loss_tuple.generator_loss,
            estimator=estimator,
        )

    @cached_property
    def _discriminator(self) -> Discriminator:
        return discriminator_factory.create(self.args, self.meta_data)

    @classmethod
    def model_args(cls):
        return discriminator_factory.MODEL_ARGS

    @classmethod
    def objective_args(cls):
        return GAN_ARGS

    @classmethod
    def regularizer_args(cls):
        return [discriminator_factory.REGULARIZER_ARG]

    @classmethod
    def optimizer_args(cls):
        return [D_OPTIMIZER_ARG]


D_OPTIMIZER_ARG = create_optimizer_action_of('discriminator')
GAN_ARGS = [
    create_action(
        '--loss',
        type=LookUp({
            'alt': GANLossTuple(lambda fake_score: BCE(fake_score, labels=1.)),  # RKL - 2JS
            'JS': GANLossTuple(lambda fake_score: -BCE(fake_score, labels=0.)),  # 2JS
            'KL': GANLossTuple(lambda fake_score: -torch.exp(fake_score)),  # -sig / (1 - sig)
            'RKL': GANLossTuple(lambda fake_score: -fake_score),  # log((1 - sig) / sig)
        }),
        default='RKL',
        help='loss function pair of GAN.',
    ),
    create_factory_action(
        '--estimator',
        type=LookUpCall({
            'reinforce': ReinforceEstimator,
            'st': StraightThroughEstimator,
            'taylor': TaylorEstimator,
            'gumbel': GumbelSoftmaxEstimator,
        }),
        default='taylor',
        help_prefix="gradient estimator for discrete sampling.\n",
    ),
    create_action(
        '--d-steps',
        type=IntRange(minval=1),
        default=1,
        help='update generator every n discriminator steps.',
    ),
]
