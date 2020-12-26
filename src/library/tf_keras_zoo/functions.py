import numpy as np
import tensorflow as tf


def takewhile_mask(condition: tf.Tensor, exclusive: bool = True, dtype=tf.bool) -> tf.Tensor:
    mask = tf.cast(condition, tf.int8)
    mask = tf.cumprod(mask, axis=1, exclusive=exclusive)
    mask = tf.cast(mask, dtype)
    return mask


def masked_reduce(x: tf.Tensor, mask: tf.Tensor, keep_batch: bool = False) -> tf.Tensor:
    if x.shape.ndims > 2:
        x = tf.reduce_sum(x, axis=list(range(2, x.shape.ndims)))
    seq_sum = tf.reduce_sum(x * tf.cast(mask, x.dtype), axis=-1)  # shape (N)
    if keep_batch:
        return seq_sum
    return tf.reduce_mean(seq_sum)


def compute_advantage(reward, decay=0.9):
    ema = tf.train.ExponentialMovingAverage(decay=decay, zero_debias=True)
    mean_reward = tf.reduce_mean(reward)
    update_baseline = ema.apply([mean_reward])
    baseline = ema.average(mean_reward)
    with tf.control_dependencies([update_baseline]):
        return reward - baseline


def random_choice_by_logits(logits, return_gumbel: bool = False, dtype=tf.int32):
    """Sample random variable from gumbel distribution by inverse transform sampling.
    Reference:
        1. Gumbel distribution: https://en.wikipedia.org/wiki/Gumbel_distribution
        2. Inverse Tranform Sampling: https://en.wikipedia.org/wiki/Inverse_transform_sampling

    """
    eps = tf.keras.backend.epsilon()
    cdf = tf.random_uniform(
        tf.shape(logits),
        minval=eps,  # avoid log(0)
        maxval=1. - eps,  # avoid log(1) = 0 for the outer log
    )
    gumbel_var = -tf.log(-tf.log(cdf))
    outputs = tf.argmax(logits + gumbel_var, axis=-1, output_type=dtype)
    return (outputs, gumbel_var) if return_gumbel else outputs


def suffix_sum(inputs, discount: float, exclusive: bool = False):
    if discount == 1.:
        return tf.cumsum(inputs, axis=-1, exclusive=exclusive, reverse=True)
    elif discount == 0.:
        return inputs if not exclusive else tf.zeros_like(inputs)

    discount_matrix = _get_discount_matrix(
        maxlen=inputs.shape[1].value,
        discount=discount,
        exclusive=exclusive,
    )
    discounted_sum = tf.matmul(inputs, discount_matrix)  # (N, T)
    return discounted_sum


def suffix_sum_with_gradients(inputs, xs, discount: float = 0.5):
    '''
    Assume `reward` is causal on `xs` along axis 1,
    namely, reward[:, t] doesn't use the value of xs[:, t'] for all t < t'
    '''
    discount_matrix = _get_discount_matrix(maxlen=inputs.shape[1].value, discount=discount)
    discounted_sum = tf.matmul(inputs, discount_matrix)  # (N, T)
    grad, = tf.gradients(discounted_sum[:, 0], xs)  # (N, T, E)
    grad /= discount_matrix[:, 0, tf.newaxis]  # (T, 1)
    return discounted_sum, grad


def _get_discount_matrix(maxlen, discount, exclusive=False):
    i_minus_j = np.arange(maxlen)[:, np.newaxis] - np.arange(maxlen)
    discount_matrix = tf.constant(
        np.tril(np.power(discount, i_minus_j), k=0 if not exclusive else -1),
        dtype=tf.float32,
    )  # (T, T)
    return discount_matrix


def pairwise_euclidean(embeddings):
    square_term = tf.reduce_sum(tf.square(embeddings), axis=-1)  # (V, )
    dot_term = tf.matmul(embeddings, embeddings, transpose_b=True)  # (V, V)
    square_dist = square_term[:, tf.newaxis] - 2 * dot_term + square_term[tf.newaxis]  # (V, V)
    return square_dist


def gaussian(square_distance):
    return tf.exp(-0.5 * square_distance)
