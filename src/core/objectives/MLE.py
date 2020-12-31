from .collections import LossCollection


class MLEObjective:

    def __call__(self, generator, real_samples):
        samples = generator.teacher_forcing_generate(real_samples)
        NLL = samples.seq_neg_logprobs.mean()
        return LossCollection(NLL, NLL=NLL)
