import numpy as np
import torch

from ..discrete import TaylorEstimator


def test_taylor_first_order():
    x0 = torch.rand((3, 4)).requires_grad_()
    y = torch.sin(x0)
    xs = torch.rand((5, 4))

    with torch.no_grad():
        first_order_y = TaylorEstimator.taylor_first_order(y, x0, xs)
        expected_output = torch.matmul(
            xs.unsqueeze(0) - x0.unsqueeze(1),
            torch.cos(x0).unsqueeze(2),
        ).squeeze(-1)

    np.testing.assert_array_almost_equal(first_order_y, expected_output)
