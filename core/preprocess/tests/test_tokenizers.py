import pytest

from ..tokenizers import Tokenizer, UttutTokenizer


class TokenizerTestTemplate:

    def test_mapping_consistent(self, tokenizer, corpus_config):
        with open(corpus_config.path.train, 'r') as f:
            line = f.readline()
            ids1 = tokenizer.text_to_ids(line)
            text1 = tokenizer.ids_to_text(ids1)
            ids2 = tokenizer.text_to_ids(text1)
            text2 = tokenizer.ids_to_text(ids2)

        assert text1 == text2
        assert ids1 == ids2

    def test_save_load(self, tokenizer, tmpdir):
        path = tmpdir / 'tokenizer.json'
        tokenizer.save(path)
        self.assert_equal(tokenizer, Tokenizer.load(path))


class TestUttutTokenizer(TokenizerTestTemplate):

    @pytest.fixture(scope='class')
    def tokenizer(self, corpus_config):
        return UttutTokenizer.fit_corpus(corpus_config)

    def assert_equal(self, a, b):
        assert a.tokens == b.tokens
        assert a.maxlen == b.maxlen
