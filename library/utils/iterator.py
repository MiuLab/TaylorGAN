from contextlib import contextmanager

import numpy as np
from tqdm import tqdm

from .file_helper import count_lines


@contextmanager
def tqdm_open(filepath, mode='r'):
    total = count_lines(filepath)
    with open(filepath, mode) as f:
        yield tqdm(f, total=total, unit='line')


def batch_generator(data, batch_size: int, shuffle: bool = False, full_batch_only: bool = False):
    total = len(data)
    stop = (total - batch_size + 1) if full_batch_only else total
    if shuffle:
        ids = np.random.permutation(total)
        for start in range(0, stop, batch_size):
            batch_ids = ids[start: start + batch_size]
            yield data[batch_ids]
    else:
        for start in range(0, stop, batch_size):
            yield data[start: start + batch_size]
