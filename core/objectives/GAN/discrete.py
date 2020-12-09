import abc

import tensorflow as tf

from library.tf_keras_zoo.functions import (
    compute_advantage,
    gaussian,
    masked_reduce,
    pairwise_euclidean,
)
from core.objectives.collections import LossCollection

from .base import GANObjective


class DiscreteGANObjective(GANObjective):

    def __call__(self, generator, real_samples):
        fake_samples = generator.generate(real_samples.batch_size, real_samples.maxlen)
        fake_result = self.discriminator.score_samples(fake_samples)

        adv_loss = self.generator_loss(fake_result.score)
        policy_loss = self._compute_loss(adv_loss, fake_result)
        return LossCollection(policy_loss, adv=tf.reduce_mean(adv_loss))

    @abc.abstractmethod
    def _compute_loss(self, adv_loss, fake_result):
        pass


class ReinforceGAN(DiscreteGANObjective):

    def __init__(self, discriminator, generator_loss, baseline_decay: float = 0.9):
        super().__init__(discriminator, generator_loss)
        self.baseline_decay = baseline_decay

    def _compute_loss(self, adv_loss, fake_result):
        reward = -tf.squeeze(adv_loss, axis=1)  # shape (N, )
        advantage = compute_advantage(reward, decay=self.baseline_decay)  # shape (N, )
        return tf.reduce_mean(tf.stop_gradient(advantage) * fake_result.seq_neg_logprobs)


class TaylorGAN(DiscreteGANObjective):

    def __init__(
            self,
            discriminator,
            generator_loss,
            baseline_decay: float = 0.9,
            bandwidth: float = 0.5,
        ):
        super().__init__(discriminator, generator_loss)
        self.baseline_decay = baseline_decay
        self.bandwidth = bandwidth

    def _compute_loss(self, adv_loss, fake_result):
        reward = -adv_loss
        first_order_reward = self.taylor_first_order(
            y=reward,
            x0=fake_result.word_vecs,
            xs=self.discriminator.embeddings,
        )
        zeroth_order_advantage = compute_advantage(reward, decay=self.baseline_decay)
        advantage = zeroth_order_advantage[:, :, tf.newaxis] + first_order_reward

        square_dist = pairwise_euclidean(self.discriminator.embeddings)
        kernel = gaussian(square_dist / (self.bandwidth ** 2))  # (V, V)
        batch_kernel = tf.nn.embedding_lookup(kernel, fake_result.ids)  # shape (N, T, V)
        likelihood = tf.tensordot(fake_result.probs, kernel, axes=[2, 0])

        normalized_advantage = batch_kernel * advantage / (likelihood + tf.keras.backend.epsilon())
        full_loss = -tf.stop_gradient(normalized_advantage) * fake_result.probs
        return masked_reduce(full_loss, mask=fake_result.mask)

    @staticmethod
    def taylor_first_order(y, x0, xs):
        dy, = tf.gradients(y, x0)  # (N, T, E)
        return (
            tf.tensordot(dy, xs, axes=[-1, -1])  # (N, T, V)
            - tf.reduce_sum(dy * x0, axis=-1, keepdims=True)  # (N, T, 1)
        )


class StraightThroughSoftmaxGAN(DiscreteGANObjective):

    def _compute_loss(self, adv_loss, fake_result):
        d_word_vecs, = tf.gradients(adv_loss, fake_result.word_vecs)  # (N, T, E)
        # NOTE, can be derived by chain-rule
        d_onehot = tf.tensordot(d_word_vecs, self.discriminator.embeddings, axes=[-1, -1])
        full_loss = tf.stop_gradient(d_onehot) * fake_result.probs  # (N, T, V)
        return masked_reduce(full_loss, mask=fake_result.mask)


class GumbelStraightThroughSoftmaxGAN(DiscreteGANObjective):

    def _compute_loss(self, adv_loss, fake_result):
        d_word_vecs, = tf.gradients(adv_loss, fake_result.word_vecs)  # (N, T, E)
        # NOTE, can be derived by chain-rule
        d_onehot = tf.tensordot(d_word_vecs, self.discriminator.embeddings, axes=[-1, -1])
        gumbel_softmax = tf.nn.softmax(fake_result.logits + fake_result.gumbel_vars)
        full_loss = tf.stop_gradient(d_onehot) * gumbel_softmax  # (N, T, V)
        return masked_reduce(full_loss, mask=fake_result.mask)
