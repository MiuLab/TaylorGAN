import pytest

import numpy as np
import tensorflow as tf

from ..model import Model
from ..sequential import Sequential


class ModelSupportMask(Model):

    def __init__(self):
        super().__init__()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        return inputs * tf.cast(mask, inputs.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask):
        return mask


@pytest.fixture(params=[
    ModelSupportMask(),
    Sequential([ModelSupportMask()]),
], scope='module')
def network(request):
    return request.param


def test_support_mask_args_used(network, sess):
    outputs = network(
        tf.constant([1., 2., 3., 4., 5.]),
        mask=tf.constant([True, False, True, True, False]),
    )
    # It won't propagate explicit mask argument
    # but still use it in 'call'
    np.testing.assert_array_almost_equal(
        sess.run(outputs),
        np.array([1., 0., 3., 4., 0.]),
    )


def test_support_input_mask_propagate(network, sess):
    inputs = tf.constant([[1., 2., 3., 4., 5.]])
    masked_inputs = tf.keras.layers.Masking()(inputs)
    outputs = network(masked_inputs)
    assert outputs._keras_mask is not None
