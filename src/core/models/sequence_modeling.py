import tensorflow as tf
import torch as th

from library.torch_zoo.functions import takewhile_mask, random_choice_by_logits
from library.utils import cached_property


class TokenSequence:

    def __init__(self, ids: th.Tensor, eos_idx: int = None, pad_idx: int = None):
        if eos_idx is not None:
            self.mask = takewhile_mask(tf.not_equal(ids, eos_idx))
            if pad_idx is not None:
                pad_idx_tensor = th.full_like(ids, pad_idx)
                ids = th.where(self.mask, ids, pad_idx_tensor)
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
    def length(self) -> tf.Tensor:
        if self.mask is None:
            return self.maxlen
        return tf.reduce_sum(tf.cast(self.mask, tf.int32), axis=1)


class SampledTokenSequence(TokenSequence):

    def __init__(
            self,
            logits: th.Tensor,
            ids: th.Tensor = None,
            gumbel_vars: th.Tensor = None,
            eos_idx: int = None,
            pad_idx: int = None,
        ):
        if ids is None:
            ids, gumbel_vars = random_choice_by_logits(logits, return_gumbel=True)

        super().__init__(ids, eos_idx=eos_idx, pad_idx=pad_idx)
        self.logits = logits
        self.gumbel_vars = gumbel_vars

    @property
    def vocab_size(self):
        return self.logits.shape[-1]

    @cached_property
    def probs(self):
        return th.nn.softmax(self.logits, dim=-1)

    @cached_property
    def neg_logprobs(self):
        return th.nn.functional.cross_entropy(
            self.logits.view(-1, self.vocab_size),
            target=self.ids.view(-1),
        ).view_like(self.ids)  # (N, T)

    @cached_property
    def seq_neg_logprobs(self):
        return th.sum(
            self.neg_logprobs * self.mask.type_as(self.neg_logprobs),
            dim=-1,
        )  # (N, )
