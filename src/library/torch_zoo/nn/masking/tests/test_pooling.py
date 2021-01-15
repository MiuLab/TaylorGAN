import pytest

import numpy as np
import torch

from ....functions import sequence_mask
from ..pooling import MaskAvgPool1d, MaskGlobalAvgPool1d


@pytest.mark.parametrize('padding', [1, 0])
@pytest.mark.parametrize('count_include_pad', [True, False])
def test_mask_avg_pool1d(padding, count_include_pad):
    with torch.no_grad():
        x = torch.arange(10, dtype=torch.float32)[None, None].expand(2, 1, 10)  # shape (2, 1, 10)
        mask = sequence_mask(torch.tensor([4, 9]), maxlen=10)
        layer = MaskAvgPool1d(kernel_size=3, padding=padding, count_include_pad=count_include_pad)
        output = layer(x, mask)
        output_mask = layer.compute_mask(mask)

    if padding == 1:
        # digit: True, x: False, p: pad, () each pooling window
        # (p01)(23x)(xxx)(xxp)  seqlen = 4
        # (p01)(234)(567)(8xp)  seqlen = 9
        expected_out = [
            [[avg(0, 1), avg(2, 3), 0., 0.]],
            [[avg(0, 1), avg(2, 3, 4), avg(5, 6, 7), 8.]],
        ]
        expected_mask = [
            [True, True, False, False],
            [True, True, True, False],  # True if at least half is True
        ]
    else:
        # digit: True, F: False, p: pad, () each pooling window
        # (012)(3xx)(xxx)x  seqlen = 4
        # (012)(345)(678)x  seqlen = 9
        expected_out = [
            [[avg(0, 1, 2), 3., 0.]],
            [[avg(0, 1, 2), avg(3, 4, 5), avg(6, 7, 8)]],
        ]
        expected_mask = [
            [True, False, False],
            [True, True, True],  # True if all is True
        ]

    np.testing.assert_array_almost_equal(output, expected_out, decimal=4)
    np.testing.assert_array_equal(output_mask, expected_mask)


def test_mask_global_avg_pool_1d():
    with torch.no_grad():
        x = torch.arange(10)[None, None].expand(2, 1, 10)  # shape (2, 1, 10)
        mask = sequence_mask(torch.tensor([4, 9]), maxlen=10)
        layer = MaskGlobalAvgPool1d(dim=2)
        out = layer(x, mask)

    expected_out = [
        [avg(*range(4))],
        [avg(*range(9))],
    ]
    np.testing.assert_array_almost_equal(out, expected_out)


def avg(*args):
    return sum(args) / len(args)
