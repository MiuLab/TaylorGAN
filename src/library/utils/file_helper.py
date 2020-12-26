from more_itertools import ilen, with_iter


def count_lines(filepath):
    return ilen(with_iter(open(filepath)))
