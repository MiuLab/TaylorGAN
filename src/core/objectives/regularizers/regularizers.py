import tensorflow as tf

from core.models import Generator, Discriminator
from library.tf_keras_zoo.functions import compute_advantage, masked_reduce

from .base import Regularizer, LossCollection


class WordVectorRegularizer(Regularizer):

    loss_name = 'word_vec'

    def __init__(self, coeff: float, max_norm: float = 0.):
        super().__init__(coeff)
        self.max_norm = max_norm

    def compute_loss(self, discriminator: Discriminator, real_samples, fake_samples):
        real_vecs = discriminator.get_embedding(real_samples.ids)
        fake_vecs = discriminator.get_embedding(fake_samples.ids)
        real_L2_loss = tf.reduce_sum(tf.square(real_vecs), axis=-1)  # shape (N, T)
        fake_L2_loss = tf.reduce_sum(tf.square(fake_vecs), axis=-1)  # shape (N, T)
        if self.max_norm:
            real_L2_loss = tf.maximum(real_L2_loss - self.max_norm ** 2, 0.)
            fake_L2_loss = tf.maximum(fake_L2_loss - self.max_norm ** 2, 0.)
        return tf.reduce_mean(real_L2_loss + fake_L2_loss) / 2


class GradientPenaltyRegularizer(Regularizer):

    loss_name = 'grad_penalty'

    def __init__(self, coeff: float, center: float = 1.):
        super().__init__(coeff)
        self.center = center

    def compute_loss(self, discriminator: Discriminator, real_samples, fake_samples):
        real_vecs = discriminator.get_embedding(real_samples.ids)
        fake_vecs = discriminator.get_embedding(fake_samples.ids)
        eps = tf.random_uniform(shape=[real_vecs.shape[0].value, 1, 1], minval=0., maxval=1.)
        inter_word_vecs = real_vecs * eps + fake_vecs * (1 - eps)
        score = discriminator.score_word_vector(inter_word_vecs)

        d_word_vecs, = tf.gradients(score, inter_word_vecs)  # (N, T, E)
        grad_norm = tf.norm(d_word_vecs, axis=[1, 2])  # (N, )
        return tf.reduce_mean(tf.square(grad_norm - self.center))


class EntropyRegularizer(Regularizer):

    loss_name = 'entropy'

    def __init__(self, coeff: float, implementation: str = 'dense'):
        super().__init__(coeff)
        self.implementation = implementation

    def __call__(self, generator: Generator, real_samples) -> LossCollection:
        fake_samples = generator.generate(real_samples.batch_size, real_samples.maxlen)
        loss = self.compute_loss(fake_samples)
        entropy = tf.reduce_mean(fake_samples.seq_neg_logprobs)
        return LossCollection(self.coeff * loss, entropy=entropy)

    def compute_loss(self, fake_samples):
        if self.implementation == 'sparse':
            NLL = fake_samples.seq_neg_logprobs
            advantage = compute_advantage(NLL)  # higher is better
            return tf.reduce_mean(tf.stop_gradient(advantage) * NLL)
        elif self.implementation == 'dense':
            # NOTE it's biased
            logp = tf.nn.log_softmax(fake_samples.logits)  # (N, T, V)
            neg_entropy = tf.reduce_sum(
                tf.stop_gradient(logp) * fake_samples.probs,
                axis=-1,
            )  # (N, T)
            return masked_reduce(neg_entropy, mask=fake_samples.mask)  # scalar
        elif self.implementation == 'chainrule':
            logp = tf.nn.log_softmax(fake_samples.logits)  # (N, T, V)
            entropy = tf.reduce_sum(
                -tf.stop_gradient(logp) * fake_samples.probs,
                axis=-1,
            )  # (N, T)
            entropy = masked_reduce(entropy, mask=fake_samples.mask, keep_batch=True)  # (N, )
            advantage = compute_advantage(entropy)  # higher is better
            sparse_part = tf.stop_gradient(advantage) * fake_samples.seq_neg_logprobs
            dense_part = -entropy
            return tf.reduce_mean(sparse_part + dense_part)  # scalar
        else:
            raise KeyError
