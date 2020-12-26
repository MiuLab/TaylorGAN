import numpy as np
import tensorflow as tf

from ..look_ahead import LookAhead


def test_look_ahead(sess):
    alpha, lr = 0.2, 0.1
    explore_steps = 5
    slow_val, grad_val = 1., 2.
    opt = LookAhead(
        tf.train.GradientDescentOptimizer(lr),
        alpha=alpha,
        explore_steps=explore_steps,
    )
    with tf.variable_scope('test_look_ahead'):
        x = tf.Variable(slow_val)
        update_x = opt.minimize(grad_val * x)  # constant grad

    sess.run(tf.variables_initializer(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='test_look_ahead'),
    ))

    for _ in range(5):
        fast_val = slow_val
        for _ in range(explore_steps - 1):
            sess.run(update_x)
            fast_val -= lr * grad_val

        np.testing.assert_almost_equal(sess.run(x), fast_val)

        sess.run(update_x)
        fast_val -= lr * grad_val

        # step % explore_steps == 0, fast interpolates with slow
        x_val = sess.run(x)
        np.testing.assert_almost_equal(
            x_val,
            slow_val * (1 - alpha) + fast_val * alpha,
        )
        slow_val = x_val
