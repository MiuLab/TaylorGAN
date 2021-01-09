import pytest
import tensorflow as tf


@pytest.fixture(scope='session')
def sess():
    with tf.Session() as sess:
        yield sess
