import torch as th


def takewhile_mask(condition: th.Tensor, exclusive: bool = True) -> th.Tensor:
    if exclusive:
        return th.cat(
            [th.ones([condition.shape[0], 1]), th.cummin(condition[:, :-1], dim=1)[0]],
            dim=1,
        ).bool()
    else:
        return th.cummin(condition, dim=1)[0].bool()


def masked_reduce(x: th.Tensor, mask: th.Tensor, keep_batch: bool = False) -> th.Tensor:
    if x.ndim > 2:
        x = th.sum(x, axis=list(range(2, x.shape.ndims)))
    seq_sum = th.sum(x * mask, axis=-1)  # shape (N)
    if keep_batch:
        return seq_sum
    return th.reduce_mean(seq_sum)


def random_choice_by_logits(logits, return_gumbel: bool = False, eps: float = 1e-8):
    """Sample random variable from gumbel distribution by inverse transform sampling.
    Reference:
        1. Gumbel distribution: https://en.wikipedia.org/wiki/Gumbel_distribution
        2. Inverse Tranform Sampling: https://en.wikipedia.org/wiki/Inverse_transform_sampling

    """
    cdf = th.clamp(
        th.rand_like(logits),
        min=eps,  # avoid log(0)
        max=1. - eps,  # avoid log(1) = 0 for the outer log
    )
    gumbel_var = -th.log(-th.log(cdf))
    outputs = th.argmax(logits + gumbel_var, dim=-1)
    return (outputs, gumbel_var) if return_gumbel else outputs


def pairwise_euclidean(embeddings):
    square_term = th.sum(embeddings ** 2, dim=-1)  # (V, )
    dot_term = th.matmul(embeddings, embeddings.T)  # (V, V)
    return square_term.unsqueeze(dim=1) - 2 * dot_term + square_term.unsqueeze(dim=0)  # (V, V)


def gaussian(square_distance):
    return th.exp(-0.5 * square_distance)
