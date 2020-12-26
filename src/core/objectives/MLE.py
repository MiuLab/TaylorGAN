import tensorflow as tf

from .collections import LossCollection


class MLEObjective:

    def __call__(self, generator, real_samples):
        samples = generator.teacher_forcing_generate(real_samples)
        NLL = tf.reduce_mean(samples.seq_neg_logprobs)
        return LossCollection(NLL, NLL=NLL)
