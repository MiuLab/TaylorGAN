import pytest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from ..pooling import (
    MaskAveragePooling1D,
    MaskMaxPooling1D,
    MaskGlobalAveragePooling1D,
    MaskGlobalMaxPooling1D,
)


@pytest.mark.parametrize('padding', ['same', 'valid'])
def test_mask_avg_pool1d(padding, sess):
    layer = MaskAveragePooling1D(pool_size=3, padding=padding)
    x = tf.constant([
        list(range(10)),
        list(range(10)),
    ], dtype=tf.float32)[:, :, tf.newaxis]  # shape (2, 10, 1)
    x._keras_mask = tf.sequence_mask([4, 9], maxlen=10)
    out = layer(x)

    def avg(*args):
        return sum(args) / len(args)

    if padding == 'same':
        # digit: True, x: False, p: pad, () each pooling window
        # (p01)(23x)(xxx)(xxp)  seqlen = 4
        # (p01)(234)(567)(8xp)  seqlen = 9
        expected_out = [
            [avg(0, 1), avg(2, 3), 0., 0.],
            [avg(0, 1), avg(2, 3, 4), avg(5, 6, 7), 8.],
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
            [avg(0, 1, 2), 3., 0.],
            [avg(0, 1, 2), avg(3, 4, 5), avg(6, 7, 8)],
        ]
        expected_mask = [
            [True, False, False],
            [True, True, True],  # True if all is True
        ]

    expected_out = np.array(expected_out)[:, :, np.newaxis]
    np.testing.assert_array_almost_equal(sess.run(out), expected_out, decimal=4)
    np.testing.assert_array_equal(sess.run(out._keras_mask), expected_mask)


@pytest.mark.parametrize('padding', ['same', 'valid'])
def test_mask_max_pool1d(padding, sess):
    layer = MaskMaxPooling1D(pool_size=3, padding=padding)
    x = tf.constant([
        list(range(10)),
        list(range(10)),
    ], dtype=tf.float32)[:, :, tf.newaxis]  # shape (2, 10, 1)
    x._keras_mask = tf.sequence_mask([4, 9], maxlen=10)
    out = layer(x)

    if padding == 'same':
        # digit: True, x: False, p: pad, () each pooling window
        # (p01)(23x)(xxx)(xxp)  seqlen = 4
        # (p01)(234)(567)(8xp)  seqlen = 9
        expected_out = [
            [1., 3., 7 - 1e4, 9 - 1e4],
            [1., 4., 7., 8.],
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
            [2., 3., 8. - 1e4],
            [2., 5., 8.],
        ]
        expected_mask = [
            [True, False, False],
            [True, True, True],  # True if all is True
        ]

    expected_out = np.array(expected_out)[:, :, np.newaxis]
    np.testing.assert_array_almost_equal(sess.run(out), expected_out, decimal=4)
    np.testing.assert_array_equal(sess.run(out._keras_mask), expected_mask)


class TestMaskGlobalAveragePooling1D:

    @pytest.fixture(scope='class')
    def inputs(self):
        return tf.placeholder(tf.float32, [None, 5, 1])

    @pytest.fixture(scope='class')
    def mask(self, inputs):
        maxlen = inputs.shape[1].value
        return tf.placeholder(tf.bool, [None, maxlen])

    @pytest.fixture(scope='class')
    def layer(self):
        return MaskGlobalAveragePooling1D()

    def test_mask_value(self, sess, inputs, mask, layer):
        outputs = layer(inputs, mask=mask)

        inputs_val = np.array([
            [0., 1., 2., 3., 4.],
            [2., 3., 4., 5., 6.],
        ]).reshape(-1, 5, 1)

        sess.run(tf.variables_initializer(var_list=layer.variables))
        outputs_val = sess.run(
            outputs,
            feed_dict={
                inputs: inputs_val,
                mask: [
                    [True, True, False, True, False],
                    [False, False, False, False, False],
                ],
            },
        )

        expected_outputs_val = np.array([(0. + 1. + 3.) / 3., 0.]).reshape(-1, 1)
        np.testing.assert_array_almost_equal(outputs_val, expected_outputs_val)

    def test_support_masked_inputs(self, inputs, layer):
        masked_inputs = tf.keras.layers.Masking()(inputs)
        with patch.object(layer, 'call') as mock_layer_call:
            layer(masked_inputs)

        kwargs = mock_layer_call.call_args[1]
        assert isinstance(kwargs['mask'], tf.Tensor)  # has passed to layer


def test_mask_global_max_pool1d(sess):
    layer = MaskGlobalMaxPooling1D()
    x = tf.constant([
        [1., 2., 3., 4., 5.],
        [1., 2., 3., 4., 5.],
    ])[:, :, tf.newaxis]  # shape (2, 5, 1)
    x._keras_mask = tf.sequence_mask([3, 0], maxlen=5)
    out = layer(x)
    np.testing.assert_array_almost_equal(
        sess.run(out),
        np.array([[3.], [0.]]),
    )
