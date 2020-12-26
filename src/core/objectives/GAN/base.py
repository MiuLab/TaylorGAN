import abc

from typing import Callable

import tensorflow as tf

from core.objectives.collections import LossCollection
from library.utils import FormatableMixin


class GANObjective(FormatableMixin):

    def __init__(self, discriminator, generator_loss, estimator):
        self.discriminator = discriminator
        self.generator_loss = generator_loss
        self.estimator = estimator

    def __call__(self, generator, real_samples):
        fake_samples = generator.generate(real_samples.batch_size, real_samples.maxlen)
        return self.estimator.compute_loss(
            fake_samples,
            discriminator=self.discriminator,
            generator_loss=self.generator_loss,
        )

    def get_config(self):
        return {'estimator': self.estimator}


class GANEstimator(abc.ABC, FormatableMixin):

    @abc.abstractmethod
    def compute_loss(self, fake_samples, discriminator, generator_loss):
        pass


class GANLossTuple:

    def __init__(
            self,
            generator_loss: Callable[[tf.Tensor], tf.Tensor],
            discriminator_loss: Callable[[tf.Tensor, tf.Tensor], tf.Tensor] = None,
        ):
        self.generator_loss = generator_loss
        self._discriminator_loss = discriminator_loss or D_BCE

    def discriminator_loss(self, discriminator, real_samples, fake_samples):
        loss = self._discriminator_loss(
            real_score=discriminator.score_samples(real_samples),
            fake_score=discriminator.score_samples(fake_samples),
        )
        return LossCollection(loss, adv=loss)


def D_BCE(real_score, fake_score):
    loss_real = BCE(real_score, labels=1.)
    loss_fake = BCE(fake_score, labels=0.)
    return tf.reduce_mean(loss_real + loss_fake)


def BCE(score, labels):
    return tf.nn.sigmoid_cross_entropy_with_logits(
        logits=score,
        labels=tf.fill(score.shape, value=labels),
    )
