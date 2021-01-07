from core.models import Generator
from core.models.sequence_modeling import TokenSequence

from .collections import LossCollection


class MLEObjective:

    def __call__(self, generator: Generator, real_samples: TokenSequence):
        NLL = generator.seq_neg_logprobs(real_samples.ids).mean()
        return LossCollection(NLL, NLL=NLL)
