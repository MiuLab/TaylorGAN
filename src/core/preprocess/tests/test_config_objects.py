from ..config_objects import CorpusConfig, Namespace


def test_is_valid(language_config, data_dir):
    assert CorpusConfig(
        path=data_dir / 'train.txt',
        language_config=language_config,
    ).is_valid


def test_isnot_valid(language_config, data_dir):
    assert not CorpusConfig(
        path=None,
        language_config=language_config,
    ).is_valid()
    assert not CorpusConfig(
        path='some.fucking.not.exist.file',
        language_config=language_config,
    ).is_valid()
    assert not CorpusConfig(
        path=Namespace(garbage=data_dir / 'train.txt'),
        language_config=language_config,
    ).is_valid()
