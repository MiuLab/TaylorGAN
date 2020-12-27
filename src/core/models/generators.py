import abc

import tensorflow as tf
import torch as th
from torch.nn import Module

from core.preprocess import SpecialTokenConfig
from library.torch_zoo.functions import random_choice_by_logits

from .interfaces import ModuleInterface
from .sequence_modeling import TokenSequence, SampledTokenSequence


class Generator(ModuleInterface):

    scope = 'Generator'

    def __init__(
            self,
            embedder: Module,
            special_token_config: SpecialTokenConfig,
            name: str = None,
        ):
        super().__init__(name)
        self.embedder = embedder
        self.special_token_config = special_token_config

    @abc.abstractmethod
    def generate(self, batch_size, maxlen, temperature):
        pass

    @property
    def embedding_matrix(self):
        return self.embedder.weight


class AutoRegressiveGenerator(Generator):

    def __init__(
            self,
            cell: Module,
            embedder: Module,
            output_layer: Module,
            special_token_config: SpecialTokenConfig,
            name: str = None,
        ):
        super().__init__(embedder, special_token_config, name)
        self.cell = cell
        self.output_layer = output_layer

    def generate(
            self,
            batch_size: int,
            maxlen: int,
            temperature: float = None,
        ) -> SampledTokenSequence:
        word_idx, state = self._get_start_token_and_state(batch_size)
        logits_list, ids_list, gv_list = [], [], []

        for _ in range(maxlen):
            word_logits, state = self._step_func(word_idx, state)
            if temperature is not None:
                word_logits /= temperature
            word_idx, gv = random_choice_by_logits(word_logits, return_gumbel=True)

            logits_list.append(word_logits)
            ids_list.append(word_idx)
            gv_list.append(gv)

        return SampledTokenSequence(
            logits=th.stack(logits_list, axis=1),
            ids=th.stack(ids_list, axis=1),
            gumbel_vars=th.stack(gv_list, axis=1),
            eos_idx=self.special_token_config.eos.idx,
            pad_idx=self.special_token_config.pad.idx,
        )

    def teacher_forcing_generate(self, samples: TokenSequence) -> SampledTokenSequence:
        sos_idx, state = self._get_start_token_and_state(batch_size=tf.shape(samples.ids)[0])
        word_ids_with_sos = [sos_idx] + tf.unstack(samples.ids, axis=1)[:-1]
        logits_list = []
        for word_idx in word_ids_with_sos:
            word_logits, state = self._step_func(word_idx, state)
            logits_list.append(word_logits)

        return SampledTokenSequence(
            logits=tf.stack(logits_list, axis=1),
            ids=samples.ids,
            eos_idx=self.special_token_config.eos.idx,
            pad_idx=self.special_token_config.pad.idx,
        )

    def _get_start_token_and_state(self, batch_size):
        sos_idx = th.full([batch_size], self.special_token_config.sos.idx)
        state = th.zeros([batch_size, self.cell.hidden_size])
        return sos_idx, state

    def _step_func(self, word_idx, state):
        word_vec = self.embedder(word_idx)
        # TODO LSTM
        output = state = self.cell(word_vec, state)  # output shape (N, C)
        word_logits = self.output_layer(output)
        return word_logits, state

    @property
    def networks(self):
        return [self.cell, self.embedder, self.output_layer]
