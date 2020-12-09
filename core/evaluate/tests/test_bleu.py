import numpy as np

from ..bleu import (
    get_closest_values,
    hashable_ngrams,
    BLEUCalculator,
    SmoothingFunction,
)


EOS = -1


def test_get_closest_values():
    ref_lengths = np.array([1, 3, 6, 10])
    np.testing.assert_array_equal(
        get_closest_values(np.arange(11, dtype=ref_lengths.dtype), target=ref_lengths),
        # 0 1  2  3  4  5  6  7  8   9  10
        [1, 1, 1, 3, 3, 6, 6, 6, 6, 10, 10],  # will get the lower if tie
    )


def test_hashable_ngrams():
    s = np.array([1, 2, 3, 1, 2, 3, 4])
    assert len(set(hashable_ngrams(s, 1))) == 4  # 1, 2, 3, 4
    assert len(set(hashable_ngrams(s, 2))) == 4  # 12, 23, 31, 34
    assert len(set(hashable_ngrams(s, 3))) == 4  # 123, 231, 312, 234
    assert len(set(hashable_ngrams(s, 4))) == 4  # 1231, 2312, 3123, 1234
    assert len(set(hashable_ngrams(s, 5))) == 3  # 12312, 23123, 31234


def test_trivial():
    ref = [
        [0, 1, 2, EOS, 0],   # seqlen = 3
        [0, 1, 1, 2, EOS],   # seqlen = 4
    ]
    cand = ref
    calculator = BLEUCalculator(ref, eos_idx=EOS, max_gram=5)
    np.testing.assert_array_almost_equal(
        calculator.bleu(cand),
        [
            [1, 1, 1, 0, 0],  # seqlen = 3
            [1, 1, 1, 1, 0],  # seqlen = 4
        ],
    )


def test_with_clipped():
    ref = [
        [0, 1, 2, 3, 4],  # has 1 `1`
        [0, 1, 1, 2, 3],  # has 2 `1`
    ]
    cand = [[1, 1, 1, 1, 1]]  # has 5 `1`
    calculator = BLEUCalculator(ref, eos_idx=EOS, max_gram=5)
    np.testing.assert_array_almost_equal(
        calculator._modified_precision(cand),
        [[2 / 5, 1 / 4, 0, 0, 0]],
        # min(2, 5) / 5, min(1, 4) / 4
    )
    np.testing.assert_array_almost_equal(
        calculator.bleu(cand),
        [[2 / 5, (2 / 5 * 1 / 4) ** 0.5, 0, 0, 0]],
    )


def test_long():
    ref = [[0, 1, 2, 0, 1, 2, 3, 0, 1, 2]]
    cand = [[0, 1, 0, 1, 2, 1, 0, 1, 2, 3]]
    calculator = BLEUCalculator(ref, eos_idx=EOS, max_gram=4)
    np.testing.assert_array_almost_equal(
        calculator._modified_precision(cand),
        [[
            9 / 10,  # 4 `1` in cand, clipped by 3 `1` in ref
            6 / 9,   # 2 `10` + 1 `21` not in ref
            3 / 8,   # 2 `012` + 1 `123`
            1 / 7,   # 1 `0123`
        ]],
    )


def test_with_brevity_penalty():
    ref = [
        [0, 1, 2, EOS, 0, 0, 0, 0, 0, 0],  # L = 3
        [0, 1, 2, 3, 4, EOS, 0, 0, 0, 0],  # L = 5
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # L = 10
    ]
    cand = [
        [0, 1, EOS, 0, 0, 0, 0, 0],  # L = 2
        [0, 1, 2, 3, EOS, 0, 0, 0],  # L = 4
        [0, 1, 2, 3, 4, 5, 6, EOS],  # L = 7
        [0, 1, 2, 3, 4, 5, 6, 7],    # L = 8
    ]
    calculator = BLEUCalculator(ref, eos_idx=EOS, max_gram=1)
    np.testing.assert_array_almost_equal(
        calculator.bleu(cand),
        [
            [np.exp(1 - 3 / 2)],
            [1],  # 4 is closest to 3 & 5, choose 3 since it's tied and no penalty applied
            [1],  # 7 is closest to 5, which is shorter so no penalty applied
            [np.exp(1 - 10 / 8)],
        ],
    )


def test_smoothing():
    ref = [[0, 1, 2, 3, 4]]
    smoothing = SmoothingFunction.fuzz_smoothing
    calculator = BLEUCalculator(ref, eos_idx=EOS, max_gram=4, smoothing=smoothing)
    np.testing.assert_array_almost_equal(
        calculator.bleu([[1, 2, 0, 3, 4]]),
        [[
            1.,
            (1 * 2 / 4) ** (1 / 2),
            (1 * 2 / 4 * 0.1 / 3) ** (1 / 3),  # no matching 3-gram
            (1 * 2 / 4 * 0.1 / 3 * 0.1 / 2) ** (1 / 4),  # no matching 3/4-gram
        ]],
    )
