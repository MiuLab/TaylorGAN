import os
import pathlib
from collections import Counter

import pytest
import tensorflow as tf


def pytest_addoption(parser):
    parser.addoption(
        "--gpu",
        action="store_true",
        help="include gpu-only tests.",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--gpu"):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # disable gpu
    if not tf.test.is_gpu_available():
        # --gpu given in cli: do not skip gpu-only tests
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="need gpu to run"))


@pytest.fixture(scope='session')
def data_dir():
    return pathlib.Path(__file__).parent / 'datasets'


@pytest.fixture(scope='session')
def sess():
    with tf.Session() as sess:
        yield sess
