import abc
from typing import Callable

import tensorflow as tf

from core.objectives.collections import LossCollection


class GANObjective(abc.ABC):

    def __init__(self, discriminator, generator_loss):
        self.discriminator = discriminator
        self.generator_loss = generator_loss

    @abc.abstractmethod
    def __call__(self):
        pass

    def __str__(self):
        return f"{self.__class__.__name__}()"


class GANLossTuple:

    def __init__(
            self,
            generator_loss: Callable[[tf.Tensor], tf.Tensor],
            discriminator_loss: Callable[[tf.Tensor, tf.Tensor], tf.Tensor] = None,
        ):
        self.generator_loss = generator_loss
        self._discriminator_loss = discriminator_loss or D_BCE

    def discriminator_loss(self, discriminator, real_samples, fake_samples):
        real_score = discriminator.score_samples(real_samples).score
        fake_score = discriminator.score_samples(fake_samples).score
        loss = self._discriminator_loss(real_score, fake_score)
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
