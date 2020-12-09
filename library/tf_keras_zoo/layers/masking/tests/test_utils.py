import numpy as np
import tensorflow as tf

from ..utils import apply_mask


def test_apply_mask(sess):
    x = tf.constant([[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]], dtype=tf.float32)
    assert apply_mask(x, None) is x

    mask = tf.constant([[True, False, True, True, False]])
    out = apply_mask(x, mask)

    np.testing.assert_array_equal(
        sess.run(out),
        [[[0., 1.], [0., 0.], [4., 5.], [6., 7.], [0., 0.]]],
    )


def test_apply_mask_just_do_once():
    x = tf.ones([3, 5])
    mask = tf.ones([3, 5], dtype=tf.bool)
    masked_x = apply_mask(x, mask)

    assert apply_mask(masked_x, mask) is masked_x
