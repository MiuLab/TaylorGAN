import os
import pathlib
from collections import Counter

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--gpu",
        action="store_true",
        help="include gpu-only tests.",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--gpu"):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # disable gpu


@pytest.fixture(scope='session')
def data_dir():
    return pathlib.Path(__file__).parent / 'datasets'
