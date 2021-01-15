import pytest

import numpy as np
import torch

from ....functions import sequence_mask
from ..mask_conv import MaskConv1d


@pytest.mark.parametrize('padding', [1, 0])
def test_mask_conv(padding):
    x = torch.ones([2, 3, 5], dtype=torch.float32)
    with torch.no_grad():
        mask = sequence_mask(torch.tensor([2, 4]), maxlen=5)
        layer = MaskConv1d(3, 1, kernel_size=3, padding=padding, bias=False)
        layer.weight.data.fill_(1.)
        output = layer(x, mask)
        output_mask = layer.compute_mask(mask)

    if padding == 1:
        # digit: True, x: False, p: pad, () each conv window
        # (p33)xxxp, p(33x)xxp, p3(3xx)xp, p33(xxx)p, p33x(xxp)
        # (p33)33xp, p(333)3xp, p3(333)xp, p33(33x)p, p333(3xp)
        expected_out = [
            [[6, 6, 3, 0, 0]],
            [[6, 9, 9, 6, 3]],
        ]
        # False if less than half are True in window
        expected_mask = [
            [True, True, False, False, False],  # act like maxlen = 2
            [True, True, True, True, False],  # act like maxlen = 4
        ]
    elif padding == 0:
        # (33x)xx, 3(3xx)x, 33(xxx)
        # (333)3x, 3(333)x, 33(33x)
        expected_out = [
            [[6, 3, 0]],
            [[9, 9, 6]],
        ]
        # False if any False in window
        expected_mask = [
            [False, False, False],  # act like maxlen = 2
            [True, True, False],  # act like maxlen = 4
        ]

    np.testing.assert_array_almost_equal(output, expected_out, decimal=4)
    np.testing.assert_array_equal(output_mask, expected_mask)
