import abc
from collections import Counter
from itertools import chain, takewhile
from more_itertools import take
from typing import Iterable, List

import numpy as np
from more_itertools import unique_everseen

from uttut.pipeline.ops import Pad, Token2Index
from uttut.pipeline.ops.add_end_token import AddEndToken

from library.utils import logging_block, JSONSerializableMixin

from .adaptors import UttutPipeline
from .config_objects import CorpusConfig, LanguageConfig, SpecialTokenConfig


class Tokenizer(abc.ABC, JSONSerializableMixin):

    special_token_config = SpecialTokenConfig(
        sos="<sos>",
        eos="</s>",
        pad="<pad>",
        unk="<unk>",
    )
    INT_DTYPE = np.int32
    eos_idx = INT_DTYPE(special_token_config.eos.idx)

    def __init__(self, tokens: List[int], maxlen: int):
        self.tokens = tokens
        self.maxlen = maxlen

    def texts_to_array(self, texts: Iterable[str]) -> np.ndarray:
        return np.array(list(map(self.text_to_ids, texts)), dtype=self.INT_DTYPE)

    @abc.abstractmethod
    def text_to_ids(self, text: str) -> List[int]:
        pass

    @abc.abstractmethod
    def ids_to_text(self, ids: List[int], split: str = None) -> str:
        pass

    @property
    def vocab_size(self):
        return len(self.tokens)

    def summary(self):
        with logging_block(f"{self.__class__.__name__} summary:"):
            print(f"Maxlen: {self.maxlen}.")
            print(f"Vocabulary size: {self.vocab_size}.")
            self.special_token_config.summary()


class UttutTokenizer(Tokenizer):

    def __init__(self, tokens: List[int], language_config: LanguageConfig, maxlen: int):
        super().__init__(tokens, maxlen)
        self.language_config = language_config
        self._token_indexer = UttutPipeline([
            AddEndToken(self.special_token_config.eos.token),
            Pad(maxlen, pad_token=self.special_token_config.pad.token),
            Token2Index(
                {token: i for i, token in enumerate(tokens)},
                unk_token=self.special_token_config.unk.token,
            ),
        ])

    def text_to_ids(self, text: str):
        tokens = self.language_config.segmentize_text(text)
        return self._token_indexer.transform_sequence(tokens)

    def ids_to_text(self, ids: List[int], split: str = None) -> str:
        if split is None:
            split = self.language_config.split_token
        tokens = [self.tokens[idx] for idx in takewhile(lambda x: x != self.eos_idx, ids)]
        return split.join(tokens)

    @classmethod
    def fit_corpus(cls, corpus_config: CorpusConfig, maxlen: int = None, vocab_size: int = None):
        maxlen = maxlen or corpus_config.maxlen
        if maxlen:
            token_freq = Counter(chain.from_iterable(
                take(maxlen, sen)
                for sen in corpus_config.iter_train_sentences()
            ))
        else:  # both preprocessor and tokenizer maxlen are None
            token_freq, maxlen = get_freq_and_maxlen(corpus_config.iter_train_sentences())

        all_tokens = unique_everseen(chain(
            cls.special_token_config.tokens,
            [token for token, _ in token_freq.most_common()],
        ))
        return cls(
            tokens=take(n=vocab_size or corpus_config.vocab_size, iterable=all_tokens),
            language_config=corpus_config.language_config,
            maxlen=maxlen,
        )

    def get_config(self):
        return {
            'tokens': self.tokens,
            'language_config': self.language_config.get_config(),
            'maxlen': self.maxlen,
        }

    @classmethod
    def from_config(cls, config_dict):
        return cls(
            tokens=config_dict['tokens'],
            language_config=LanguageConfig.from_config(config_dict['language_config']),
            maxlen=config_dict['maxlen'],
        )


def get_freq_and_maxlen(sentences):
    freq = Counter()
    maxlen = 0
    for sen in sentences:
        freq += Counter(sen)
        maxlen = max(len(sen), maxlen)
    return freq, maxlen
