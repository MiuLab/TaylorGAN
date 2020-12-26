import tensorflow as tf

from core.objectives.collections import LossCollection
from .base import GANObjective


def soft_embed(discriminator, probs):
    if not discriminator.embedder.built:
        with tf.keras.backend.name_scope(discriminator.scope):
            discriminator.embedder.build(probs.shape[:2])
        assert discriminator.embedder.built

    return tf.tensordot(probs, discriminator.embeddings, axes=[-1, 0])  # (N, T, E)


class SoftmaxGAN(GANObjective):

    def __call__(self, generator, real_samples):
        fake_samples = generator.generate(real_samples.batch_size, real_samples.maxlen)
        word_vecs = soft_embed(self.discriminator, fake_samples.probs)  # (N, T, E)
        score = self.discriminator.score_word_vector(word_vecs, fake_samples.mask)
        adv_loss = tf.reduce_mean(self.generator_loss(score))
        return LossCollection(adv_loss, adv=adv_loss)


class GumbelSoftmaxGAN(GANObjective):

    def __call__(self, generator, real_samples):
        fake_samples = generator.generate(real_samples.batch_size, real_samples.maxlen)
        gumbel_softmax = tf.nn.softmax(fake_samples.logits + fake_samples.gumbel_vars)
        word_vecs = soft_embed(self.discriminator, gumbel_softmax)  # (N, T, E)
        score = self.discriminator.score_word_vector(word_vecs, fake_samples.mask)
        adv_loss = tf.reduce_mean(self.generator_loss(score))
        return LossCollection(adv_loss, adv=adv_loss)
