from ..config_objects import CorpusConfig, Namespace


def test_is_valid(language_config, data_dir):
    assert CorpusConfig(
        name='test',
        path=data_dir / 'train.txt',
        language_config=language_config,
    ).is_valid


def test_isnot_valid(language_config, data_dir):
    assert not CorpusConfig(
        name='invalid path',
        path=None,
        language_config=language_config,
    ).is_valid()
    assert not CorpusConfig(
        name='file_not_exist',
        path='some.fucking.not.exist.file',
        language_config=language_config,
    ).is_valid()
    assert not CorpusConfig(
        name='no_training_set',
        path=Namespace(garbage=data_dir / 'train.txt'),
        language_config=language_config,
    ).is_valid()
