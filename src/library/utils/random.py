import numpy as np


def random_sample(arr, size: int):
    if size > len(arr):
        raise ValueError(f"expect `size` <= length of `arr`, Found {size} > {len(arr)}!")
    elif size == len(arr):
        return arr
    else:
        sample_ids = np.random.choice(len(arr), replace=False, size=[size])
        return arr[sample_ids]
