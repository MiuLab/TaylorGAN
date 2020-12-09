import pytest

import numpy as np
import tensorflow as tf

from ..weight_decay import WeightDecay


def test_weight_decay(sess):
    lr, decay_rate = 0.2, 0.1
    x_val, z_val = 2., 1.
    optimizer = WeightDecay(
        tf.train.GradientDescentOptimizer(lr),
        decay_rate=decay_rate,
    )
    x = tf.Variable(x_val)
    z = tf.Variable(z_val)
    y = tf.pow(x, 3)  # dy/dx = 3x^2
    train_op = optimizer.minimize(y, var_list=[x, z])

    sess.run(tf.variables_initializer([x, z]))
    sess.run(train_op)
    np.testing.assert_almost_equal(
        sess.run(x),
        x_val * (1. - decay_rate) - lr * 3 * (x_val ** 2),
    )
    np.testing.assert_almost_equal(sess.run(z), z_val)  # keep since it's not updated


@pytest.mark.parametrize('var_filter', ['collection', 'callable'])
def test_weight_decay_with_filter(var_filter, sess):
    lr, decay_rate = 0.2, 0.1
    x_val, z_val = 2., 1.
    x = tf.Variable(x_val, name='x')
    z = tf.Variable(z_val, name='z')

    optimizer = WeightDecay(
        tf.train.GradientDescentOptimizer(lr),
        decay_rate=decay_rate,
        variable_filter={x} if var_filter == 'collection' else lambda v: 'x' in v.name,
    )
    y = tf.pow(x, 3) + z  # dy/dx = 3x^2, dy/dz = 1
    train_op = optimizer.minimize(y, var_list=[x, z])

    sess.run(tf.variables_initializer([x, z]))
    sess.run(train_op)
    np.testing.assert_almost_equal(
        sess.run(x),
        x_val * (1. - decay_rate) - lr * 3 * (x_val ** 2),
    )
    np.testing.assert_almost_equal(
        sess.run(z),
        z_val - lr,
    )  # doesn't decay since it's not in filter


@pytest.mark.parametrize('sparse_update', [True, False])
def test_sparse_weight_decay(sparse_update, sess):
    lr, decay_rate = 0.2, 0.1
    E_val = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    x = tf.constant([[0, 1, 1]])
    E = tf.Variable(E_val, dtype=tf.float32, name='E')

    optimizer = WeightDecay(
        tf.train.GradientDescentOptimizer(lr),
        decay_rate=decay_rate,
        sparse_update=sparse_update,
    )
    e = tf.nn.embedding_lookup(E, x)
    y = tf.pow(e, 3)  # dy/de = 3e^2
    train_op = optimizer.minimize(y, var_list=[E])

    sess.run(E.initializer)
    sess.run(train_op)
    if sparse_update:
        expected_E_val = [
            E_val[0] * (1 - decay_rate) - lr * (3 * E_val[0] ** 2),  # occurrence 1
            E_val[1] * (1 - 2 * decay_rate) - 2 * lr * (3 * E_val[1] ** 2),  # occurrence 2
            E_val[2],
        ]
    else:
        expected_E_val = [
            E_val[0] * (1 - decay_rate) - lr * (3 * E_val[0] ** 2),  # occurrence 1
            E_val[1] * (1 - decay_rate) - 2 * lr * (3 * E_val[1] ** 2),  # occurrence 2
            E_val[2] * (1 - decay_rate),
        ]

    np.testing.assert_array_almost_equal(sess.run(E), expected_E_val, decimal=4)
