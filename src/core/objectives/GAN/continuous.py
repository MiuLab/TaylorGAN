import torch

from core.objectives.collections import LossCollection

from .base import GANEstimator


class SoftmaxEstimator(GANEstimator):

    def compute_loss(self, fake_samples, discriminator, generator_loss):
        return _compute_loss_of_probability(
            discriminator,
            generator_loss,
            probs=fake_samples.probs,
            mask=fake_samples.mask,
        )


class GumbelSoftmaxEstimator(GANEstimator):

    def compute_loss(self, fake_samples, discriminator, generator_loss):
        return _compute_loss_of_probability(
            discriminator,
            generator_loss,
            probs=torch.nn.functional.softmax(fake_samples.logits + fake_samples.gumbel_vars),
            mask=fake_samples.mask,
        )


def _compute_loss_of_probability(discriminator, generator_loss, probs, mask):
    word_vecs = torch.tensordot(probs, discriminator.embedding_matrix, dims=1)  # (N, T, E)
    score = discriminator.score_word_vector(word_vecs, mask)
    adv_loss = generator_loss(score).mean()
    return LossCollection(adv_loss, adv=adv_loss)
