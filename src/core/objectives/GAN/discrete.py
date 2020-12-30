import torch

from library.tf_keras_zoo.functions import (
    compute_advantage,
    gaussian,
    masked_reduce,
    pairwise_euclidean,
)
from core.models.discriminators import Discriminator
from core.models.sequence_modeling import SampledTokenSequence
from core.objectives.collections import LossCollection

from .base import GANEstimator


class ReinforceEstimator(GANEstimator):

    def __init__(self, baseline_decay: float = 0.9):
        self.baseline_decay = baseline_decay
        self.baseline = None

    def compute_loss(
            self,
            fake_samples: SampledTokenSequence,
            discriminator: Discriminator,
            generator_loss: callable,
        ):
        score = discriminator.score_samples(fake_samples)
        adv_loss = generator_loss(score)
        reward = adv_loss.squeeze(axis=1)  # shape (N, )

        advantage = self.compute_advantage(reward)  # shape (N, )
        policy_loss = (advantage.detach() * fake_samples.seq_neg_logprobs).mean()
        return LossCollection(policy_loss, adv=adv_loss.mean())

    def compute_advantage(self, reward):
        if self.baseline is None:
            self.baseline = reward.mean()

        advantage = reward - self.baseline
        self.baseline = (
            self.baseline * self.baseline_decay + reward.mean() * (1 - self.baseline_decay)
        )
        return advantage


class TaylorEstimator(GANEstimator):

    def __init__(self, baseline_decay: float = 0.9, bandwidth: float = 0.5):
        self.baseline_decay = baseline_decay
        self.bandwidth = bandwidth

    def compute_loss(
            self,
            fake_samples: SampledTokenSequence,
            discriminator: Discriminator,
            generator_loss: callable,
        ):
        fake_embeddings = discriminator.get_embedding(word_ids=fake_samples.ids)
        score = discriminator.score_word_vector(fake_embeddings)
        adv_loss = generator_loss(score)
        reward = -adv_loss

        first_order_reward = self.taylor_first_order(
            y=reward,
            x0=fake_embeddings,
            xs=discriminator.embedding_matrix,
        )
        zeroth_order_advantage = compute_advantage(reward, decay=self.baseline_decay)
        advantage = zeroth_order_advantage.unsqueeze(dim=2) + first_order_reward

        square_dist = pairwise_euclidean(discriminator.embedding_matrix)
        kernel = gaussian(square_dist / (self.bandwidth ** 2))  # (V, V)
        batch_kernel = tf.nn.embedding_lookup(kernel, fake_samples.ids)  # shape (N, T, V)
        likelihood = torch.tensordot(fake_samples.probs, kernel, axes=[2, 0])

        normalized_advantage = batch_kernel * advantage / (likelihood + 1e-8)
        full_loss = -normalized_advantage.detach() * fake_samples.probs
        policy_loss = masked_reduce(full_loss, mask=fake_samples.mask)
        return LossCollection(policy_loss, adv=adv_loss.mean())

    @staticmethod
    def taylor_first_order(y, x0, xs):
        dy, = tf.gradients(y, x0)  # (N, T, E)
        return (
            torch.tensordot(dy, xs, axes=[-1, -1]) - (dy * x0).sum(-1).unsqueeze(dim=2)
        )


class StraightThroughEstimator(GANEstimator):

    def compute_loss(
            self,
            fake_samples: SampledTokenSequence,
            discriminator: Discriminator,
            generator_loss: callable,
        ):
        word_vecs = discriminator.get_embedding(word_ids=fake_samples.ids)
        score = discriminator.score_word_vector(word_vecs)
        adv_loss = generator_loss(score)

        d_word_vecs, = tf.gradients(adv_loss, word_vecs)  # (N, T, E)
        # NOTE, can be derived by chain-rule
        d_onehot = tf.tensordot(d_word_vecs, discriminator.embedding_matrix, axes=[-1, -1])
        full_loss = tf.stop_gradient(d_onehot) * fake_samples.probs  # (N, T, V)
        policy_loss = masked_reduce(full_loss, mask=fake_samples.mask)
        return LossCollection(policy_loss, adv=tf.reduce_mean(adv_loss))
