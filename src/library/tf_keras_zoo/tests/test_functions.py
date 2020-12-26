from collections import Counter

import numpy as np
import pytest
import tensorflow as tf

from library.tf_keras_zoo.layers import Dense, Lambda, LSTM, Conv1D
from library.tf_keras_zoo.networks import Sequential

from ..functions import (
    takewhile_mask,
    random_choice_by_logits,
    suffix_sum,
    suffix_sum_with_gradients,
)


def test_takewhile_mask(sess):
    condition = tf.constant([
        [0, 0, 1, 1, 0],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1],
    ])
    mask = takewhile_mask(condition, exclusive=True, dtype=tf.bool)
    np.testing.assert_array_almost_equal(
        mask.eval(),
        [
            [1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1],
        ],
    )
    mask = takewhile_mask(condition, exclusive=False, dtype=tf.bool)
    np.testing.assert_array_almost_equal(
        mask.eval(),
        [
            [0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ],
    )


def test_random_choice_by_logits(sess):
    N = 1000
    logits = tf.constant([1., 2., 3.])
    idx = random_choice_by_logits(logits)
    counter = Counter(idx.eval() for _ in range(N))
    probs = tf.nn.softmax(logits).eval()

    print(counter)
    # NOTE may randomly fail in a low chance.
    assert all(
        probs[i] - 0.1 < (counter[i] / N) < probs[i] + 0.1
        for i in range(3)
    )


def test_suffix_sum(sess):
    x = tf.constant([[1., 2., 3., 4., 5.]])  # shape (1, 5)
    np.testing.assert_array_equal(
        suffix_sum(x, discount=1.).eval(),
        [[15, 14, 12, 9, 5]],
    )
    np.testing.assert_array_equal(
        suffix_sum(x, discount=1., exclusive=True).eval(),
        [[14, 12, 9, 5, 0]],
    )
    np.testing.assert_array_equal(
        suffix_sum(x, discount=0.).eval(),
        [[1., 2., 3., 4., 5.]],
    )
    np.testing.assert_array_equal(
        suffix_sum(x, discount=1 / 2).eval(),
        [[
            1 + 2 / 2 + 3 / 4 + 4 / 8 + 5 / 16,
            2 + 3 / 2 + 4 / 4 + 5 / 8,
            3 + 4 / 2 + 5 / 4,
            4 + 5 / 2,
            5,
        ]],
    )
    np.testing.assert_array_equal(
        suffix_sum(x, discount=1 / 2, exclusive=True).eval(),
        [[
            2 / 2 + 3 / 4 + 4 / 8 + 5 / 16,
            3 / 2 + 4 / 4 + 5 / 8,
            4 / 2 + 5 / 4,
            5 / 2,
            0,
        ]],
    )


@pytest.mark.parametrize('model', [
    Sequential([
        LSTM(50, return_sequences=True),
        LSTM(50, return_sequences=True),
        Dense(1),
        Lambda(lambda x: tf.squeeze(x, axis=-1)),
    ]),
    Sequential([
        Conv1D(filters=10, kernel_size=3, activation='relu', padding='causal'),
        Conv1D(filters=10, kernel_size=3, activation='relu', padding='causal'),
        Dense(1),
        Lambda(lambda x: tf.squeeze(x, axis=-1)),
    ]),
])
def test_suffix_sum_with_gradients(model, sess):
    batch_size, maxlen, dim = 3, 50, 10
    inputs = tf.random_normal([batch_size, maxlen, dim])  # (N, T, D)
    reward = model(inputs)  # (N, T)
    discounted_reward, grad = suffix_sum_with_gradients(reward, inputs, discount=0.25)

    idx = np.random.choice(maxlen)
    grad_1 = grad[:, idx, :]  # (N, T, D)
    grad_2 = tf.gradients(discounted_reward[:, idx], inputs)[0][:, idx, :]  # (N, T, D)

    sess.run(tf.global_variables_initializer())
    g1, g2 = sess.run([grad_1, grad_2])

    assert np.isfinite(g1).all() and np.isfinite(g2).all()
    np.testing.assert_array_almost_equal(g1, g2)
