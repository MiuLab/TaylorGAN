import pytest
import tensorflow as tf

from ..resnet import ResBlock


@pytest.fixture(params=[ResBlock()])
def layer(request):
    return request.param


def test_output_shape(layer):
    x = tf.zeros([3, 10, 5])
    y = layer(x)
    assert y.shape.as_list() == x.shape.as_list()


def test_output_mask(layer):
    x = tf.zeros([3, 10, 5])
    x._keras_mask = tf.ones([3, 10], dtype=tf.bool)
    y = layer(x)
    assert y._keras_mask is x._keras_mask
