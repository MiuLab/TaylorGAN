from typing import List

import numpy as np
import tensorflow as tf

from core.models.sequence_modeling import TokenSequence
from core.preprocess import Tokenizer
from library.utils import batch_generator, cached_property


class TextGenerator:
    '''Facade class of Generator model'''

    def __init__(self, generator, tokenizer: Tokenizer):
        self.generator = generator
        self._tokenizer = tokenizer

    def generate_texts(self, size: int, batch_size: int = 64, temperature: float = 1.) -> List[str]:
        return list(map(
            self._tokenizer.ids_to_text,
            self.generate_ids(size, batch_size, temperature),
        ))

    def generate_ids(self, size: int, batch_size: int = 64, temperature: float = 1.) -> np.ndarray:
        batch_list = [
            self.generator.generate(
                min(size - start, batch_size),
                self._tokenizer.maxlen,
            ).ids.numpy()
            for start in range(0, max(size - batch_size, 0) + 1, batch_size)
        ]
        return np.concatenate(batch_list, axis=0)

    def ids_to_text(self, word_ids):
        return self._tokenizer.ids_to_text(word_ids)

    @classmethod
    def from_model(cls, generator, tokenizer):
        # static batch size -> build a dynamic one
        return cls(generator, tokenizer)


class PerplexityCalculator:
    '''Facade class of Generator model'''

    _WORD_IDS_KEY = 'input_place'
    _NLL_KEY = 'negative_log_likelihood'
    _SEQLEN_KEY = 'seqlen'

    def __init__(self, word_ids, NLL, seqlen):
        self._word_ids = word_ids
        self._NLL = NLL
        self._seqlen = seqlen

    def perplexity(self, inputs: np.ndarray, batch_size: int = 64) -> float:
        sess = tf.get_default_session()
        total_NLL = total_words = 0.
        for batch_x in batch_generator(inputs, batch_size, full_batch_only=False):
            batch_NLL, batch_seqlen = sess.run(
                [self._NLL, self._seqlen],
                feed_dict={self._word_ids: batch_x},
            )
            total_NLL += np.sum(batch_NLL)
            total_words += np.sum(batch_seqlen)

        avg_NLL = total_NLL / total_words
        perplexity = np.exp(avg_NLL)
        return perplexity

    @cached_property
    def signature(self):
        return tf.saved_model.signature_def_utils.predict_signature_def(
            inputs={self._WORD_IDS_KEY: self._word_ids},
            outputs={self._NLL_KEY: self._NLL, self._SEQLEN_KEY: self._seqlen},
        )

    @classmethod
    def from_signature(cls, signature):
        get_tensor = tf.get_default_graph().get_tensor_by_name
        return cls(
            word_ids=get_tensor(signature.inputs[cls._WORD_IDS_KEY].name),
            NLL=get_tensor(signature.outputs[cls._NLL_KEY].name),
            seqlen=get_tensor(signature.outputs[cls._SEQLEN_KEY].name),
        )

    @classmethod
    def from_model(cls, generator, maxlen: int):
        # static batch size -> build a dynamic one
        word_ids = tf.placeholder(tf.int32, [None, maxlen])
        samples = TokenSequence(
            word_ids,
            eos_idx=generator.special_token_config.eos.idx,
            pad_idx=generator.special_token_config.pad.idx,
        )
        return cls(
            word_ids=word_ids,
            NLL=generator.teacher_forcing_generate(samples).seq_neg_logprobs,
            seqlen=samples.length,
        )
