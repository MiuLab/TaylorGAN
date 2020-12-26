from ..format_utils import left_aligned


def test_left_aligned():
    assert left_aligned(['1', '12', '123']) == ['1  ', '12 ', '123']
