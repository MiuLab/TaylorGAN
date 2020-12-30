import torch

from core.models import Generator, Discriminator
from library.torch_zoo.functions import masked_reduce

from .base import Regularizer, LossCollection


class WordVectorRegularizer(Regularizer):

    loss_name = 'word_vec'

    def __init__(self, coeff: float, max_norm: float = 0.):
        super().__init__(coeff)
        self.max_norm = max_norm

    def compute_loss(self, discriminator: Discriminator, real_samples, fake_samples):
        real_vecs = discriminator.get_embedding(real_samples.ids)
        fake_vecs = discriminator.get_embedding(fake_samples.ids)
        real_L2_loss = torch.square(real_vecs).sum(dim=-1)  # shape (N, T)
        fake_L2_loss = torch.square(fake_vecs).sum(dim=-1)  # shape (N, T)
        if self.max_norm:
            real_L2_loss = torch.maximum(real_L2_loss - self.max_norm ** 2, 0.)
            fake_L2_loss = torch.maximum(fake_L2_loss - self.max_norm ** 2, 0.)
        return (real_L2_loss + fake_L2_loss).mean() / 2


class GradientPenaltyRegularizer(Regularizer):

    loss_name = 'grad_penalty'

    def __init__(self, coeff: float, center: float = 1.):
        super().__init__(coeff)
        self.center = center

    def compute_loss(self, discriminator: Discriminator, real_samples, fake_samples):
        real_vecs = discriminator.get_embedding(real_samples.ids)
        fake_vecs = discriminator.get_embedding(fake_samples.ids)

        eps = torch.rand(real_vecs.shape[0], 1, 1)
        inter_word_vecs = real_vecs * eps + fake_vecs * (1 - eps)
        score = discriminator.score_word_vector(inter_word_vecs)

        d_word_vecs, = torch.autograd.grad(
            score, inter_word_vecs,
            grad_outputs=torch.ones_like(score),
        )  # (N, T, E)
        grad_norm = torch.linalg.norm(d_word_vecs, dim=[1, 2])  # (N, )
        return torch.square(grad_norm - self.center).mean()


class EntropyRegularizer(Regularizer):

    loss_name = 'entropy'

    def __call__(self, generator: Generator, real_samples) -> LossCollection:
        fake_samples = generator.generate(real_samples.batch_size, real_samples.maxlen)
        loss = self.compute_loss(fake_samples)
        entropy = fake_samples.seq_neg_logprobs.mean()
        return LossCollection(self.coeff * loss, entropy=entropy)

    def compute_loss(self, fake_samples):
        # NOTE it's biased
        logp = torch.nn.functional.log_softmax(fake_samples.logits)  # (N, T, V)
        neg_entropy = (logp.detach() * fake_samples.probs).sum(dim=-1)  # (N, T)
        return masked_reduce(neg_entropy, mask=fake_samples.mask)  # scalar
