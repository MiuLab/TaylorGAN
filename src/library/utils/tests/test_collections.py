import pytest

from ..collections import counter_or, dict_of_unique, ExponentialMovingAverageMeter


def test_counter_or():
    assert counter_or([
        {'a': 3, 'b': 1},
        {'a': 2, 'c': 1},
    ]) == {'a': 3, 'b': 1, 'c': 1}


def test_dict_of_unique():
    assert dict_of_unique([(1, 2), (3, 4)]) == {1: 2, 3: 4}
    with pytest.raises(ValueError):
        dict_of_unique([(1, 2), (1, 4)])


def test_exponential_average_meter():
    meter = ExponentialMovingAverageMeter(decay=0.5)
    assert meter.apply(a=1.0) == {'a': 1.0}
    assert meter.apply(a=2.0, b=2.0) == {'a': 1.5, 'b': 2.0}
    assert meter.apply(a=3.0, b=3.0) == {'a': 2.25, 'b': 2.5}
