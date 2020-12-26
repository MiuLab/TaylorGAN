
from functools import partialmethod
from termcolor import colored
from typing import List

import numpy as np

from core.evaluate import TextGenerator
from library.utils import get_seqlens, SEPARATION_LINE, random_sample

from ..channels import channels, register_channel


class GeneratorProxy:

    def __init__(self, generator: TextGenerator):
        self.generator = generator

    def log_samples_command(self, sample_size):

        def log_samples(step, batch_data):
            print(SEPARATION_LINE)
            print()
            print(colored("Real Sentences (Random Sampled):", 'blue'))
            print_samples(map(
                self.generator.ids_to_text,
                random_sample(batch_data, sample_size),
            ))
            print()
            print(colored("Fake Sentences (Random Sampled):", 'red'))
            print_samples(self.generator.generate_texts(sample_size))
            print()

        return log_samples

    def _evaluate_command(self, evaluator: callable, sample_size: int, target_channel, method_name):
        register_channel(target_channel)

        def generate_and_evaluate(step, *_):
            samples = getattr(self.generator, method_name)(sample_size)
            vals = evaluator(samples)
            channels[target_channel].post(step, vals)

        return generate_and_evaluate

    evaluate_text = partialmethod(_evaluate_command, method_name='generate_texts')
    evaluate_ids = partialmethod(_evaluate_command, method_name='generate_ids')


def print_samples(texts: List[str]):
    for i, line in enumerate(texts, 1):
        print(f"{i}.")
        print(line)


def mean_length(word_ids, eos_idx):
    return {'mean_length': np.mean(get_seqlens(word_ids, eos_idx))}
