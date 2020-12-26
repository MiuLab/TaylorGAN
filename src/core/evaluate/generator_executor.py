import math
from typing import List

import numpy as np
import tensorflow as tf

from core.models.sequence_modeling import TokenSequence
from core.preprocess import Tokenizer
from library.utils import batch_generator, cached_property


class TextGenerator:
    '''Facade class of Generator model'''

    _BATCH_KEY = 'batch_size'
    _TEMP_KEY = 'temperature'
    _OUTPUT_KEY = 'output'

    def __init__(
            self,
            batch_size: tf.Tensor, temperature: tf.Tensor, output_ids: tf.Tensor,
            tokenizer: Tokenizer,
        ):
        self._batch_size = batch_size
        self._temperature = temperature
        self._output_ids = output_ids
        self._tokenizer = tokenizer

    def generate_texts(self, size: int, batch_size: int = 64, temperature: float = 1.) -> List[str]:
        return list(map(
            self._tokenizer.ids_to_text,
            self.generate_ids(size, batch_size, temperature),
        ))

    def generate_ids(self, size: int, batch_size: int = 64, temperature: float = 1.) -> np.ndarray:
        sess = tf.get_default_session()
        feed_dict = {
            self._batch_size: batch_size,
            self._temperature: temperature,
        }
        batch_list = [
            sess.run(self._output_ids, feed_dict=feed_dict)
            for _ in range(math.ceil(size / batch_size))
        ]
        return np.concatenate(batch_list, axis=0)[: size]

    def ids_to_text(self, word_ids):
        return self._tokenizer.ids_to_text(word_ids)

    @cached_property
    def signature(self):
        return tf.saved_model.signature_def_utils.predict_signature_def(
            inputs={self._BATCH_KEY: self._batch_size, self._TEMP_KEY: self._temperature},
            outputs={self._OUTPUT_KEY: self._output_ids},
        )

    @classmethod
    def from_signature(cls, signature, tokenizer):
        get_tensor = tf.get_default_graph().get_tensor_by_name
        return cls(
            batch_size=get_tensor(signature.inputs[cls._BATCH_KEY].name),
            temperature=get_tensor(signature.inputs[cls._TEMP_KEY].name),
            output_ids=get_tensor(signature.outputs[cls._OUTPUT_KEY].name),
            tokenizer=tokenizer,
        )

    @classmethod
    def from_model(cls, generator, tokenizer):
        # static batch size -> build a dynamic one
        batch_size = tf.placeholder_with_default(64, shape=(), name='batch_size_place')
        temperature = tf.placeholder_with_default(1., shape=(), name='temperature_place')
        output = generator.generate(batch_size, tokenizer.maxlen, temperature=temperature)
        return cls(batch_size, temperature, output.ids, tokenizer)


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
