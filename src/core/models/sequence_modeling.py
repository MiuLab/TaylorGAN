import torch

from library.torch_zoo.functions import takewhile_mask, random_choice_by_logits
from library.utils import cached_property


class TokenSequence:

    def __init__(self, ids: torch.Tensor, eos_idx: int = None, pad_idx: int = None):
        if eos_idx is not None:
            self.mask = takewhile_mask(torch.not_equal(ids, eos_idx))
            if pad_idx is not None:
                pad_idx_tensor = torch.full_like(ids, pad_idx)
                ids = torch.where(self.mask, ids, pad_idx_tensor)
        else:
            self.mask = None
        self.ids = ids

    @property
    def batch_size(self) -> int:
        return self.ids.shape[0]

    @property
    def maxlen(self) -> int:
        return self.ids.shape[1]

    @cached_property
    def length(self) -> torch.Tensor:
        if self.mask is None:
            return self.maxlen
        return self.mask.type_as(torch.int32).sum(dim=1)


class SampledTokenSequence(TokenSequence):

    def __init__(
            self,
            logits: torch.Tensor,
            ids: torch.Tensor = None,
            gumbel_vars: torch.Tensor = None,
            eos_idx: int = None,
            pad_idx: int = None,
        ):
        if ids is None:
            ids, gumbel_vars = random_choice_by_logits(logits, return_gumbel=True)

        super().__init__(ids, eos_idx=eos_idx, pad_idx=pad_idx)
        self.logits = logits
        self.gumbel_vars = gumbel_vars

    @property
    def vocab_size(self) -> int:
        return self.logits.shape[-1]

    @cached_property
    def probs(self) -> torch.Tensor:
        return torch.nn.functional.softmax(self.logits, dim=-1)

    @cached_property
    def seq_neg_logprobs(self) -> torch.Tensor:
        return seq_neg_logprobs(self.logits, self.ids, mask=self.mask)  # (N, )


def seq_neg_logprobs(logits, ids, mask=None):
    batch_size, maxlen, vocab_size = logits.shape
    neg_logprobs = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size),
        target=ids.view(-1),
        reduction='none',
    ).view(batch_size, maxlen)  # (N, T)

    if mask is not None:
        return (neg_logprobs * mask.type_as(neg_logprobs)).sum(dim=-1)  # (N, )
    else:
        return neg_logprobs.sum(dim=-1)
