import pytest

import numpy as np
import tensorflow as tf

from ..conv1d_transpose import Conv1DTranspose


@pytest.fixture(scope='module')
def inputs():
    return tf.placeholder(dtype=tf.float32, shape=[None, 3, 1])


def test_configs():
    dconv1d = Conv1DTranspose(filters=3, kernel_size=5, padding='valid')
    assert {'filters': 3, 'kernel_size': (5,)}.items() <= dconv1d.get_config().items()


@pytest.mark.parametrize('padding', ['valid', 'same'])
def test_output_shape(inputs, padding):
    width, channel = inputs.shape.as_list()[1:]
    filters, kernel_size = 3, 5
    dconv1d = Conv1DTranspose(filters=filters, kernel_size=kernel_size, padding=padding)
    outputs = dconv1d(inputs)

    if padding == 'valid':
        expected_shape = [None, width + kernel_size - 1, filters]
    elif padding == 'same':
        expected_shape = [None, width, filters]
    else:
        raise AssertionError("Invalid parametrize!")

    assert outputs.shape.as_list() \
        == dconv1d.compute_output_shape(inputs.shape).as_list() \
        == expected_shape


@pytest.mark.parametrize('padding', ['valid', 'same'])
def test_output_value(inputs, padding, sess):
    width, channel = inputs.shape.as_list()[1:]
    dconv1d = Conv1DTranspose(
        filters=1,
        kernel_size=3,
        kernel_initializer='ones',  # for manually compute output val
        padding=padding,
    )
    outputs = dconv1d(inputs)

    sess.run(tf.variables_initializer(var_list=dconv1d.variables))
    outputs_val = sess.run(
        outputs,
        feed_dict={inputs: np.array([1., 2., 3.]).reshape(-1, width, channel)},
    )

    if padding == 'valid':
        # for visualizing the deconv process
        expected_val = np.sum([
            np.array([[[1.], [1.], [1.], [0.], [0.]]]),
            np.array([[[0.], [2.], [2.], [2.], [0.]]]),
            np.array([[[0.], [0.], [3.], [3.], [3.]]]),
        ], axis=0)
    elif padding == 'same':
        expected_val = np.sum([
            [[[1.], [1.], [0.]]],
            [[[2.], [2.], [2.]]],
            [[[0.], [3.], [3.]]],
        ], axis=0)
    else:
        raise AssertionError("Invalid parametrize!")

    np.testing.assert_array_almost_equal(outputs_val, expected_val)


@pytest.mark.parametrize('invalid_inputs', [
    tf.zeros(shape=[2, 3]),
    tf.zeros(shape=[2, 3, 1, 1]),
])
def test_raise_invalid_input_rank(invalid_inputs):
    layer = Conv1DTranspose(filters=10, kernel_size=5)
    with pytest.raises(ValueError):
        layer(invalid_inputs)


def test_raise_invalid_input_channel_after_built(inputs):
    width, channel = inputs.shape.as_list()[1:]
    dconv1d = Conv1DTranspose(filters=10, kernel_size=5)
    dconv1d(inputs)  # build on this inputs channel

    different_channel_inputs = tf.placeholder(dtype=tf.float32, shape=[None, width, channel + 1])
    with pytest.raises(ValueError):
        dconv1d(different_channel_inputs)
