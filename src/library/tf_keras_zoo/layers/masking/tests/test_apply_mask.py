import tensorflow as tf

from ..apply_mask import ApplyMask


def test_apply_mask_layer_propagate_mask():
    layer = ApplyMask()
    x = tf.ones([5, 3])
    x._keras_mask = tf.ones([5, 3], dtype=tf.bool)
    out = layer(x)

    assert out._keras_mask is x._keras_mask
