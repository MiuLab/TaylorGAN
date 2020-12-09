import numpy as np

from ..array_utils import get_seqlens


def test_get_seqlens():
    array = [[1, 2, 3, 4, 0], [2, 3, 0, 1, 2], [1, 2, 3, 4, 5]]
    np.testing.assert_array_equal(
        get_seqlens(array, eos_idx=0),
        [4, 2, 5],
    )
