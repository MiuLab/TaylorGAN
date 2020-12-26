import tensorflow as tf

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
            probs=tf.nn.softmax(fake_samples.logits + fake_samples.gumbel_vars),
            mask=fake_samples.mask,
        )


def _compute_loss_of_probability(discriminator, generator_loss, probs, mask):
    if not discriminator.embedder.built:
        with tf.keras.backend.name_scope(discriminator.scope):
            discriminator.embedder.build(probs.shape[:2])
        assert discriminator.embedder.built

    word_vecs = tf.tensordot(probs, discriminator.embedding_matrix, axes=[-1, 0])  # (N, T, E)
    score = discriminator.score_word_vector(word_vecs, mask)
    adv_loss = tf.reduce_mean(generator_loss(score))
    return LossCollection(adv_loss, adv=adv_loss)
