import os
from collections import namedtuple
from typing import Callable, List, Union

from more_itertools import all_unique, with_iter
from uttut.pipeline.ops import (
    CharTokenizer,
    MergeWhiteSpaceCharacters,
    StripWhiteSpaceCharacters,
)

from library.utils import logging_indent, tqdm_open, format_path, JSONSerializableMixin

from .adaptors import UttutPipeline, WordEmbeddingCollection


class Namespace:

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()


class LanguageConfig(JSONSerializableMixin):

    def __init__(
            self,
            embedding_path: str,
            segmentor: UttutPipeline = None,
            split_token: str = '',
        ):
        self.embedding_path = embedding_path
        if segmentor is None:
            self._segmentor = UttutPipeline([
                MergeWhiteSpaceCharacters(),
                StripWhiteSpaceCharacters(),
                CharTokenizer(),
            ])
        else:
            self._segmentor = segmentor

        self.split_token = split_token

    def segmentize_text(self, text):
        return self._segmentor.transform_sequence(text)

    def load_pretrained_embeddings_msg(self):
        if not (self.embedding_path and os.path.isfile(self.embedding_path)):
            raise FileNotFoundError(f"invalid embedding_path: {self.embedding_path}")
        print(f"Load pretrained embedding from : {format_path(self.embedding_path)}")
        return WordEmbeddingCollection.load_msg(self.embedding_path)

    def get_config(self):
        return {
            'embedding_path': str(self.embedding_path),
            'segmentor': self._segmentor.serialize(),
            'split_token': self.split_token,
        }

    @classmethod
    def from_config(cls, config_dict):
        config_dict['segmentor'] = UttutPipeline.deserialize(config_dict['segmentor'])
        return cls(**config_dict)


class CorpusConfig:

    def __init__(
            self,
            path: Union[str, Namespace],
            language_config: LanguageConfig,
            maxlen: int = None,  # used when preprocessor.maxlen = None
            vocab_size: int = None,  # used when preprocessor.vocab_size = None
        ):
        self.path = path if isinstance(path, Namespace) else Namespace(train=path)
        self.language_config = language_config
        self.maxlen = maxlen
        self.vocab_size = vocab_size

    def iter_train_sentences(self, segmentize_func: Callable[[str], List[str]] = None):
        segmentize_func = segmentize_func or self.language_config.segmentize_text
        return map(segmentize_func, with_iter(tqdm_open(self.path.train)))

    def is_valid(self):
        try:
            return hasattr(self.path, 'train') and all(map(os.path.isfile, self.path.values()))
        except TypeError:
            return False


class SpecialTokenConfig:

    TokenIdxTuple = namedtuple('TokenIdxTuple', ['token', 'idx'])

    def __init__(self, **kwargs):
        if not all_unique(kwargs.values()):
            raise KeyError("special tokens conflict.")

        self._token_list = list(kwargs.values())
        self._attrs = {
            key: self.TokenIdxTuple(token, idx)
            for idx, (key, token) in enumerate(kwargs.items())
        }

    @property
    def tokens(self):
        return self._token_list

    def __getattr__(self, key):
        if key in self._attrs:
            return self._attrs[key]
        return super().__getattribute__(key)

    def summary(self):
        with logging_indent("Special tokens config:"):
            for key, (token, idx) in self._attrs.items():
                print(f"{key} token: '{token}', index: {idx}.")
