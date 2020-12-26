import numpy as np

from core.cache import cache_center
from library.utils import logging_indent

from .config_objects import LanguageConfig, Namespace


class DataCollection(Namespace):

    def summary(self):
        with logging_indent("Data summary:"):
            for key, array in self.items():
                print(f"{key} data contains {len(array)} sentences.")


class TextDataset:

    def __init__(self, ids, texts):
        self.ids = ids
        self.texts = texts

    def __len__(self):
        return len(self.ids)


class MetaData:

    def __init__(self, tokenizer, corpus_config, cache_dir):
        self.tokenizer = tokenizer
        self.corpus_config = corpus_config
        self.cache_dir = cache_dir

    def load_pretrained_embeddings(self) -> np.ndarray:

        @cache_center.to_npz(self.cache_dir / 'word_vecs.npz')
        def load_embeddings():
            word_vec_config = self.language_config.load_pretrained_embeddings_msg()
            return word_vec_config.get_matrix_of_tokens(self.tokenizer.tokens)

        with logging_indent("Load pretrained embeddings:"):
            embeddings = load_embeddings()
            print(f"Dimensions: {embeddings.shape[1]}.")

        return embeddings

    def summary(self):
        self.tokenizer.summary()

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def maxlen(self) -> int:
        return self.tokenizer.maxlen

    @property
    def eos_idx(self) -> int:
        return self.tokenizer.eos_idx

    @property
    def special_token_config(self):
        return self.tokenizer.special_token_config

    @property
    def language_config(self) -> LanguageConfig:
        return self.corpus_config.language_config
