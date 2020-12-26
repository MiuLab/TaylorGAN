from ..preprocessors import UttutPreprocessor


def test_uttut_preprocessor(corpus_config):
    data_collection = UttutPreprocessor().preprocess(corpus_config)
    assert data_collection.train.ids.shape[1] == corpus_config.maxlen
