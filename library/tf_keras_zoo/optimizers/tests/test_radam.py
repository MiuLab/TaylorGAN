import numpy as np
import tensorflow as tf

from ..radam import RAdamOptimizer


def test_radam(sess):
    lr = 0.1
    radam_opt = RAdamOptimizer(lr)
    with tf.variable_scope('test_radam'):
        x = tf.Variable(1.)
        update_x = radam_opt.minimize(2 * x)  # constant grad 2

    sess.run(tf.variables_initializer(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='test_radam'),
    ))
    for _ in range(4):
        sess.run(update_x)

    x_val = sess.run(x)
    np.testing.assert_almost_equal(x_val, 1. - 4 * lr * 2)  # without adaptive gradient

    # N_sma > 4 now
    rectifier, _ = sess.run([radam_opt.rectifier, update_x])
    new_x_val = sess.run(x)
    np.testing.assert_almost_equal(
        new_x_val,
        x_val - lr * rectifier * 2 / 2,  # with adaptive gradient: divide by sqrt(v)
    )
