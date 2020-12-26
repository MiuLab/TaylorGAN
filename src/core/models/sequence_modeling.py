import tensorflow as tf

from library.tf_keras_zoo.functions import takewhile_mask, random_choice_by_logits, masked_reduce
from library.utils import cached_property


class TokenSequence:

    def __init__(self, ids, eos_idx: int = None, pad_idx: int = None):
        if eos_idx is not None:
            self.mask = takewhile_mask(tf.not_equal(ids, eos_idx))
            if pad_idx is not None:
                pad_idx_tensor = tf.fill(tf.shape(ids), pad_idx)
                ids = tf.where(self.mask, ids, pad_idx_tensor)
        else:
            self.mask = None
        self.ids = ids

    @property
    def batch_size(self) -> int:
        return self.ids.shape[0].value

    @property
    def maxlen(self) -> int:
        return self.ids.shape[1].value

    @cached_property
    def length(self) -> tf.Tensor:
        if self.mask is None:
            return self.ids.shape[1].value
        return tf.reduce_sum(tf.cast(self.mask, tf.int32), axis=1)


class SampledTokenSequence(TokenSequence):

    def __init__(self, logits, ids=None, gumbel_vars=None, eos_idx=None, pad_idx=None):
        if ids is None:
            ids, gumbel_vars = random_choice_by_logits(logits, return_gumbel=True)

        super().__init__(ids, eos_idx=eos_idx, pad_idx=pad_idx)
        self.logits = logits
        self.probs = tf.nn.softmax(logits)
        self.gumbel_vars = gumbel_vars

    @property
    def vocab_size(self):
        return self.logits.shape[-1].value

    @cached_property
    def neg_logprobs(self):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits,
            labels=self.ids,
        )  # (N, T)

    @cached_property
    def seq_neg_logprobs(self):
        return masked_reduce(self.neg_logprobs, self.mask, keep_batch=True)  # (N, )
