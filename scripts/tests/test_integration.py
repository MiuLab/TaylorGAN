import gc

import pytest
import tensorflow as tf

from core.cache import cache_center
from .. import (
    train,
    restore_from_checkpoint,
    generate_text,
    perplexity,
    evaluate_text,
)


@pytest.fixture(scope='session')
def cache_root_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('cache')


@pytest.fixture(scope='session')
def serving_root(tmpdir_factory):
    return tmpdir_factory.mktemp('tf_serving')


@pytest.fixture(scope='session')
def checkpoint_root(tmpdir_factory):
    return tmpdir_factory.mktemp('checkpoint')


@pytest.fixture(scope='session', autouse=True)
def redirect_cache_root(cache_root_dir):
    cache_center.root_path = str(cache_root_dir)


@pytest.fixture(autouse=True)
def transact():
    yield
    tf.reset_default_graph()
    gc.collect()


class TestTrain:

    @pytest.mark.dependency(name='train_GAN')
    def test_GAN(self, serving_root, checkpoint_root):
        argv = ' '.join([
            '--data test',
            '-g test -d test --estimator taylor',
            '--g-op sgd learning_rate=1e-3,clip_value=0.1',
            '--g-reg embedding coeff=0.1',
            '--g-reg entropy coeff=1e-5',
            '--d-op sgd learning_rate=1e-3,clip_global_norm=1',
            '--d-reg grad_penalty coeff=10.',
            '--d-reg spectral coeff=0.1',
            '--d-reg embedding coeff=0.1',
            '--epochs 4 --batch 2',
            '--bleu 2',
            f'--serv {serving_root} --ckpt {checkpoint_root} --save-period 2',
        ]).split()
        train.main(argv)


class TestEvaluate:

    @pytest.mark.dependency(name='save_serving', depends=['train_GAN'])
    def test_serving_model_is_saved(self, serving_root):
        epochs, period = 4, 2
        model_dir = min(serving_root.listdir())
        assert (model_dir / 'tokenizer.json').isfile()

        for epo in range(period, epochs, period):
            epo_dirname = model_dir / f'tf_model_epo{epo}'
            assert epo_dirname.isdir()
            assert tf.saved_model.loader.maybe_saved_model_directory(epo_dirname / '0')

    @pytest.mark.dependency(name='restore', depends=['train_GAN'])
    def test_restore(self, checkpoint_root, serving_root):
        restore_path = min(checkpoint_root.listdir())
        restore_from_checkpoint.main(argv=f'{restore_path} --epochs 6 --save-period 5'.split())
        # successfully change saving_epochs
        assert tf.train.latest_checkpoint(restore_path).endswith('5')
        assert tf.saved_model.loader.maybe_saved_model_directory(
            serving_root / restore_path.basename / 'tf_model_epo5' / '0',
        )

    @pytest.mark.dependency(name='generate_text', depends=['save_serving'])
    def test_generate_text(self, tmpdir, serving_root):
        model_path = min(serving_root.listdir()) / 'tf_model_epo4'
        export_path = tmpdir / 'generated_text.txt'
        generate_text.main(
            argv=f'--model {model_path} --export {export_path} --samples 100'.split(),
        )
        assert len(export_path.readlines()) == 100

    @pytest.mark.dependency(name='perplexity', depends=['save_serving'])
    def test_perplexity(self, serving_root):
        model_path = min(serving_root.listdir()) / 'tf_model_epo4'
        perplexity.main(argv=f'--model {model_path} --data test'.split())

    def test_evaluate_text(self, data_dir):
        corpus_path = data_dir / 'train.txt'
        evaluate_text.main(argv=f'--eval {corpus_path} --data test --bleu 5'.split())
