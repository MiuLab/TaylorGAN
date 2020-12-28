import torch


def takewhile_mask(condition: torch.Tensor, exclusive: bool = True) -> torch.Tensor:
    if exclusive:
        return torch.cat(
            [torch.ones([condition.shape[0], 1]), torch.cummin(condition[:, :-1], dim=1)[0]],
            dim=1,
        ).bool()
    else:
        return torch.cummin(condition, dim=1)[0].bool()


def masked_reduce(x: torch.Tensor, mask: torch.Tensor, keep_batch: bool = False) -> torch.Tensor:
    if x.ndim > 2:
        x = torch.sum(x, axis=list(range(2, x.shape.ndims)))
    seq_sum = torch.sum(x * mask, axis=-1)  # shape (N)
    if keep_batch:
        return seq_sum
    return torch.reduce_mean(seq_sum)


def random_choice_by_logits(logits, return_gumbel: bool = False, eps: float = 1e-8):
    """Sample random variable from gumbel distribution by inverse transform sampling.
    Reference:
        1. Gumbel distribution: https://en.wikipedia.org/wiki/Gumbel_distribution
        2. Inverse Tranform Sampling: https://en.wikipedia.org/wiki/Inverse_transform_sampling

    """
    cdf = torch.clamp(
        torch.rand_like(logits),
        min=eps,  # avoid log(0)
        max=1. - eps,  # avoid log(1) = 0 for the outer log
    )
    gumbel_var = -torch.log(-torch.log(cdf))
    outputs = torch.argmax(logits + gumbel_var, dim=-1)
    return (outputs, gumbel_var) if return_gumbel else outputs


def pairwise_euclidean(embeddings):
    square_term = torch.sum(embeddings ** 2, dim=-1)  # (V, )
    dot_term = torch.matmul(embeddings, embeddings.T)  # (V, V)
    return square_term.unsqueeze(dim=1) - 2 * dot_term + square_term.unsqueeze(dim=0)  # (V, V)


def gaussian(square_distance):
    return torch.exp(-0.5 * square_distance)
