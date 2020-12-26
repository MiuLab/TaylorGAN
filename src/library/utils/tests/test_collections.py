from ..collections import counter_or


def test_counter_or():
    assert counter_or([
        {'a': 3, 'b': 1},
        {'a': 2, 'c': 1},
    ]) == {'a': 3, 'b': 1, 'c': 1}
