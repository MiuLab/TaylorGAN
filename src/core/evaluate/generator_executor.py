from typing import List

import numpy as np
import torch

from core.models import Generator
from core.preprocess import Tokenizer
from library.utils import batch_generator


class TextGenerator:
    '''Facade class of Generator model'''

    BATCH_SIZE = 64

    def __init__(self, generator: Generator, tokenizer: Tokenizer):
        self.generator = generator
        self._tokenizer = tokenizer

    def generate_texts(self, size: int, temperature: float = 1.) -> List[str]:
        return list(map(
            self._tokenizer.ids_to_text,
            self.generate_ids(size, temperature),
        ))

    def generate_ids(self, size: int, temperature: float = 1.) -> np.ndarray:
        return np.concatenate(
            [
                self.generator.forward(
                    torch.tensor(batch_size),
                    torch.tensor(self._tokenizer.maxlen),
                    torch.tensor(temperature),
                )
                for batch_size in compute_batch_size(size, self.BATCH_SIZE)
            ],
            axis=0,
        )

    def perplexity(self, inputs: np.ndarray) -> float:
        total_NLL = total_words = 0.
        with torch.no_grad():
            for batch_x in map(
                torch.from_numpy,
                batch_generator(inputs, self.BATCH_SIZE, full_batch_only=False),
            ):
                batch_NLL = self.generator.seq_neg_logprobs(batch_x)
                total_NLL += batch_NLL.sum()
                # TODO seqlen
                total_words += inputs.shape[0] * inputs.shape[1]

        avg_NLL = total_NLL / total_words
        perplexity = avg_NLL.exp().numpy()
        return perplexity

    def export_traced(self):
        inputs = {
            'forward': (
                torch.tensor(1),
                torch.tensor(self._tokenizer.maxlen),
                torch.tensor(1.),
            ),
            'seq_neg_logprobs': torch.zeros([1, self._tokenizer.maxlen], dtype=torch.int64),
        }
        return torch.jit.trace_module(self.generator._wrapped, inputs)

    def ids_to_text(self, word_ids):
        return self._tokenizer.ids_to_text(word_ids)

    @classmethod
    def load_traced(cls, path, tokenizer):
        return cls(torch.jit.load(str(path)), tokenizer)


def compute_batch_size(total_size, batch_size):
    q, m = divmod(total_size, batch_size)
    for _ in range(q):
        yield batch_size
    if m:
        yield m
