import abc
import pathlib

import numpy as np
from more_itertools import with_iter

from library.utils import logging_block, tqdm_open, format_path
from core.cache import cache_center

from .config_objects import CorpusConfig
from .record_objects import DataCollection, TextDataset, MetaData
from .tokenizers import UttutTokenizer


class Preprocessor(abc.ABC):

    @abc.abstractmethod
    def preprocess(self, corpus_config: CorpusConfig) -> MetaData:
        pass


class UttutPreprocessor(Preprocessor):

    def __init__(self, maxlen: int = None, vocab_size: int = None):
        self.maxlen = maxlen
        self.vocab_size = vocab_size

    def preprocess(self, corpus_config: CorpusConfig, return_meta: bool = False):
        with logging_block("Prepare text tokenizer..."):
            tokenizer = self._create_tokenizer(corpus_config)

        with logging_block("Preprocess text corpus..."):
            data_collection = self._process_data(tokenizer, corpus_config)

        if return_meta:
            meta_data = MetaData(
                tokenizer=tokenizer,
                corpus_config=corpus_config,
                cache_dir=self.get_cache_dir(corpus_config),
            )
            return data_collection, meta_data
        else:
            return data_collection

    def _create_tokenizer(self, corpus_config):
        @cache_center.to_json(self.get_cache_dir(corpus_config) / 'tokenizer.json')
        def create_tokenizer():
            print(
                "Build text mapper based on corpus data "
                f"from {format_path(corpus_config.path.train)}",
            )
            return UttutTokenizer.fit_corpus(
                corpus_config,
                maxlen=self.maxlen,
                vocab_size=self.vocab_size,
            )

        return create_tokenizer()

    def _process_data(self, tokenizer, corpus_config):
        data_collection = DataCollection()
        for key, path in corpus_config.path.items():
            @cache_center.to_npz(self.get_cache_dir(corpus_config) / f'{key}_data.npz')
            def _process_text_file(filepath) -> np.ndarray:
                print(f"Load corpus data from {format_path(filepath)}")
                return tokenizer.texts_to_array(with_iter(tqdm_open(filepath)))

            with logging_block(f"{key} data:", bullet=False):
                ids = _process_text_file(path)
                text = list(map(tokenizer.ids_to_text, ids))
                text_dataset = TextDataset(ids=ids, text=text)
                setattr(data_collection, key, text_dataset)

        return data_collection

    def get_cache_dir(self, corpus_config):
        items = ["uttut"]
        if self.maxlen:
            items.append(f"L{self.maxlen}")
        if self.vocab_size:
            items.append(f"V{self.vocab_size}")
        return pathlib.Path(corpus_config.name, "_".join(items))
