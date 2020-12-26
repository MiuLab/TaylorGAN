from collections import Counter, defaultdict
from typing import Callable

import numpy as np
from tqdm import tqdm

from library.utils import counter_or, get_seqlens, unpad, safe_divide
from core.cache import cache_center


class BLEUCalculator:

    INT_DTYPE = np.int32

    def __init__(
            self,
            references: np.ndarray,
            max_gram: int = 5,
            eos_idx: int = 1,
            smoothing: Callable = None,
            verbose: bool = False,
            cache_dir=None,
        ):
        self.eos_idx = self.INT_DTYPE(eos_idx)
        self.smoothing = smoothing

        references = np.asarray(references, dtype=self.INT_DTYPE)
        ref_lengths = get_seqlens(references, eos_idx=self.eos_idx)
        self.ref_counters = [
            NGramCounter(references, ref_lengths, n=n, cache_dir=cache_dir, verbose=verbose)
            for n in range(1, max_gram + 1)
        ]
        self.brevity_penalty = get_brevity_penalty_table(ref_lengths, maxlen=references.shape[1])

    def mean_bleu(self, candidates: np.ndarray) -> np.ndarray:
        mean_bleu = np.mean(self.bleu(candidates), axis=0)  # shape (max_gram)
        return {
            f"BLEU-{n}": bleu_n
            for n, bleu_n in enumerate(mean_bleu, 1)
        }

    @classmethod
    def selfbleu(cls, samples, **kwargs):
        candidates, references = np.split(samples, 2)
        bleu = cls(references, **kwargs).bleu(candidates)
        mean_bleu = np.mean(bleu, axis=0)  # shape (max_gram)
        return {
            f'SBLEU-{n}': bleu_n
            for n, bleu_n in enumerate(mean_bleu, 1)
        }

    def bleu(self, candidates: np.ndarray) -> np.ndarray:
        candidates = np.asarray(candidates, dtype=self.INT_DTYPE)
        cand_lens = get_seqlens(candidates, eos_idx=self.eos_idx)

        precisions = self._modified_precision(candidates, cand_lens)  # shape (N, max_grams)
        mean_precision = cum_geomean(precisions)  # shape (N, max_gram)
        batch_bleu = mean_precision * self.brevity_penalty[cand_lens, np.newaxis]
        return batch_bleu

    def _modified_precision(self, candidates, seqlens=None) -> np.ndarray:
        candidates = np.asarray(candidates, dtype=self.INT_DTYPE)
        if seqlens is None:
            seqlens = get_seqlens(candidates, eos_idx=self.eos_idx)

        # shape (N, max_gram)
        clipped_count = np.array([
            self._clipped_count(sen)
            for sen in unpad(candidates, seqlens)
        ])
        # Total number of n-grams = Length - (n - 1), shape (N, max_gram)
        total_count = seqlens[:, np.newaxis] - np.arange(self.max_gram)
        if self.smoothing:
            clipped_count, total_count = self.smoothing(clipped_count, total_count)
        return safe_divide(clipped_count, total_count)  # avoid zero division

    def _clipped_count(self, candidate: np.ndarray):
        return [ref_counter.clipped_count(candidate) for ref_counter in self.ref_counters]

    @property
    def max_gram(self):
        return len(self.ref_counters)


class NGramCounter:

    def __init__(self, references, seqlens, n, cache_dir=None, verbose=False):

        @cache_center.to_pkl(cache_dir, f'{n}-gram.pkl')
        def create_ngram_counter():
            seqs = unpad(references, seqlens)
            if verbose:
                print(f"Building {n}-gram table...")
                seqs = tqdm(seqs, total=len(references), unit='sample')
            return counter_or(Counter(hashable_ngrams(s, n)) for s in seqs)

        self.counter = create_ngram_counter()
        self.n = n

    def clipped_count(self, candidate):
        ref_counter, cand_counter = self.counter, defaultdict(int)
        for gram in hashable_ngrams(candidate, self.n):
            # NOTE faster than get and min
            if gram in ref_counter and cand_counter[gram] < ref_counter[gram]:
                cand_counter[gram] += 1
        return sum(cand_counter.values())


def cum_geomean(arr, axis=-1):
    return np.power(np.cumprod(arr, axis=axis), 1. / (np.arange(arr.shape[axis]) + 1))


def get_brevity_penalty_table(ref_lengths, maxlen):
    possible_lengths = np.arange(maxlen + 1)
    closest_lengths = get_closest_values(possible_lengths, target=ref_lengths)
    brevity_penalty = np.minimum(
        np.exp(1. - safe_divide(closest_lengths, possible_lengths)),
        1.,
    )
    return brevity_penalty


def get_closest_values(arr: np.ndarray, target: np.ndarray):
    # arr shape (M), target shape (N)
    target = np.unique(target)
    diff = np.abs(arr[:, np.newaxis] - target)  # shape (M, N)
    closest_ids = np.argmin(diff, axis=-1)  # shape (M), value in [0, N)
    return target[closest_ids]  # shape (M)


def hashable_ngrams(seq: np.ndarray, n: int):
    if n == 1:  # items already hashable
        return iter(seq)

    width = seq.itemsize * n
    bytes_seq = seq.tobytes()  # hashable, len = len(seq) * n
    return (
        bytes_seq[start: start + width]
        for start in range(0, len(bytes_seq) - width + 1, seq.itemsize)
    )


class SmoothingFunction:
    # Reference: http://acl2014.org/acl2014/W14-33/pdf/W14-3346.pdf

    @staticmethod  # smoothing1
    def fuzz_smoothing(numerator, denominator, eps: float = 0.1):
        numerator = np.maximum(numerator, eps)
        return numerator, denominator

    @staticmethod  # smoothing2
    def add1_smoothing(numerator, denominator):
        shift = np.ones_like(numerator)
        shift[:, 0] = 0  # don't add 1-gram
        return numerator + shift, denominator + shift
