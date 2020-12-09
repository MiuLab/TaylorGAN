import pytest

import numpy as np
import tensorflow as tf

from ..mask_conv import MaskConv1D


@pytest.mark.parametrize('padding', ['same', 'valid', 'causal'])
def test_mask_conv(padding, sess):
    layer = MaskConv1D(filters=1, kernel_size=3, kernel_initializer='ones', padding=padding)
    x = tf.ones([2, 5, 3], dtype=tf.float32)
    x._keras_mask = tf.sequence_mask([2, 4], maxlen=5)
    out = layer(x)

    sess.run(tf.variables_initializer(layer.variables))
    if padding == 'same':
        # digit: True, x: False, p: pad, () each conv window
        # (p33)xxxp, p(33x)xxp, p3(3xx)xp, p33(xxx)p, p33x(xxp)
        # (p33)33xp, p(333)3xp, p3(333)xp, p33(33x)p, p333(3xp)
        expected_out = [
            [[6.], [6.], [3.], [0.], [0.]],
            [[6.], [9.], [9.], [6.], [3.]],
        ]
        # False if less than half are True in window
        expected_mask = [
            [True, True, False, False, False],  # act like maxlen = 2
            [True, True, True, True, False],  # act like maxlen = 4
        ]
    elif padding == 'valid':
        # (33x)xx, 3(3xx)x, 33(xxx)
        # (333)3x, 3(333)x, 33(33x)
        expected_out = [
            [[6.], [3.], [0.]],
            [[9.], [9.], [6.]],
        ]
        # False if any False in window
        expected_mask = [
            [False, False, False],  # act like maxlen = 2
            [True, True, False],  # act like maxlen = 4
        ]
    elif padding == 'causal':
        # (pp3)3xxx, p(p33)xxx, pp(33x)xx, pp3(3xx)x, pp33(xxx)
        # (pp3)333x, p(p33)33x, pp(333)3x, pp3(333)x, pp33(33x)
        expected_out = [
            [[3.], [6.], [6.], [3.], [0.]],
            [[3.], [6.], [9.], [9.], [6.]],
        ]
        # False if any trailing False
        expected_mask = [
            [True, True, False, False, False],  # act like maxlen = 2
            [True, True, True, True, False],  # act like maxlen = 4
        ]

    np.testing.assert_array_almost_equal(sess.run(out), expected_out, decimal=4)
    np.testing.assert_array_equal(sess.run(out._keras_mask), expected_mask)
