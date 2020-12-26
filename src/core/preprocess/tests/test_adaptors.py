import numpy as np
import pytest
from uttut.pipeline.ops import CharTokenizer, Pad, Token2Index

from ..adaptors import UttutPipeline, WordEmbeddingCollection


class TestUttutPipeline:

    @pytest.fixture(scope='class')
    def pipe(self):
        return UttutPipeline([
            CharTokenizer(),
            Pad(20, pad_token='<pad>'),
            Token2Index(
                {'<sos>': 0, '<eos>': 1, '<unk>': 2, '<pad>': 3, 'a': 4, 'b': 5},
                unk_token='<unk>',
            ),
        ])

    def test_methods(self, pipe):
        pipe = UttutPipeline()
        assert callable(pipe.transform_sequence)
        assert callable(pipe.summary)

    def test_save_load(self, pipe, tmpdir):
        path = tmpdir / 'test_pipe.json'
        pipe.save(path)
        assert UttutPipeline.load(path) == pipe


class TestWordEmbeddingCollection:

    def test_get_matrix(self):
        wordvec = WordEmbeddingCollection(
            {'a': 0, 'b': 1, 'c': 2, WordEmbeddingCollection.UNK: 3},
            [[0, 1], [2, 3], [4, 5], [6, 7]],
        )
        assert np.array_equal(
            wordvec.get_matrix_of_tokens(['b', 'd is unk', 'a']),
            [[2, 3], [6, 7], [0, 1]],
        )
