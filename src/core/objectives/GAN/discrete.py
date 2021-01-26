import torch

from core.models.discriminators import Discriminator
from core.models.sequence_modeling import SampledTokenSequence
from core.objectives.collections import LossCollection
from library.torch_zoo.functions import gaussian, masked_reduce, pairwise_euclidean

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
        self.baseline = None

    def compute_loss(
            self,
            fake_samples: SampledTokenSequence,
            discriminator: Discriminator,
            generator_loss: callable,
        ):
        fake_embeddings = discriminator.get_embedding(word_ids=fake_samples.ids)
        score = discriminator.score_word_vector(fake_embeddings, mask=fake_samples.mask)
        adv_loss = generator_loss(score)
        reward = -adv_loss

        first_order_reward = self.taylor_first_order(
            y=reward,
            x0=fake_embeddings,
            xs=discriminator.embedding_matrix,
        ).view_as(fake_samples.logits)
        zeroth_order_advantage = self.compute_advantage(reward)
        advantage = zeroth_order_advantage.unsqueeze(dim=2) + first_order_reward

        square_dist = pairwise_euclidean(discriminator.embedding_matrix)
        kernel = gaussian(square_dist / (self.bandwidth ** 2))  # (V, V)
        batch_kernel = torch.nn.functional.embedding(fake_samples.ids, kernel)  # shape (N, T, V)
        likelihood = torch.tensordot(fake_samples.probs, kernel, dims=1)

        normalized_advantage = batch_kernel * advantage / (likelihood + 1e-8)
        full_loss = -normalized_advantage.detach() * fake_samples.probs
        policy_loss = masked_reduce(full_loss, mask=fake_samples.mask)
        return LossCollection(policy_loss, adv=adv_loss.mean())

    @staticmethod
    def taylor_first_order(y, x0, xs):
        """
        Args:
            y: any shape computed by x0
            x0: shape (*M, d)
            xs: shape (N, d)

        Returns:
            dy: shape (*M, N)
        """
        dydx0, = torch.autograd.grad(y, x0, grad_outputs=torch.ones_like(y))  # (*M, d)
        # dydx0 * (xs - x0) = dydx0 * xs - dydx0 * x0
        return (
            torch.tensordot(dydx0, xs, dims=[[-1], [-1]])  # (*M, N)
            - (dydx0 * x0).sum(dim=-1, keepdim=True)  # (*M, 1)
        )

    def compute_advantage(self, reward):
        if self.baseline is None:
            self.baseline = reward.mean()

        advantage = reward - self.baseline
        self.baseline = (
            self.baseline * self.baseline_decay + reward.mean() * (1 - self.baseline_decay)
        )
        return advantage


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

        d_word_vecs, = torch.autograd.grad(adv_loss, word_vecs)  # (N, T, E)
        # NOTE, can be derived by chain-rule
        d_onehot = torch.tensordot(d_word_vecs, discriminator.embedding_matrix, dims=[[-1], [-1]])
        full_loss = d_onehot.detach() * fake_samples.probs  # (N, T, V)
        policy_loss = masked_reduce(full_loss, mask=fake_samples.mask)
        return LossCollection(policy_loss, adv=adv_loss.mean())
