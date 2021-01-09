import numpy as np
import torch

from ..functions import takewhile_mask, random_choice_by_logits


def test_takewhile_mask():
    condition = torch.Tensor([
        [0, 0, 1, 1, 0],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1],
    ])
    np.testing.assert_array_equal(
        takewhile_mask(condition, exclusive=True),
        [
            [1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1],
        ],
    )
    np.testing.assert_array_equal(
        takewhile_mask(condition, exclusive=False),
        [
            [0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ],
    )


def test_random_choice_by_logits():
    N = 1000
    logits = torch.Tensor([1., 2., 3.])
    counter = np.bincount([
        random_choice_by_logits(logits)
        for _ in range(N)
    ])
    probs = torch.softmax(logits, dim=-1)

    print(counter)
    # NOTE may randomly fail in a low chance.
    assert all(
        probs[i] - 0.1 < (counter[i] / N) < probs[i] + 0.1
        for i in range(3)
    )
