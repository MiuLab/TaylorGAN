import numpy as np


def random_sample(arr, size: int):
    if size > len(arr):
        raise ValueError(f"expect `size` <= length of `arr`, Found {size} > {len(arr)}!")
    elif size == len(arr):
        return arr

    sample_ids = np.random.choice(len(arr), replace=False, size=[size])
    if isinstance(arr, np.ndarray):
        return arr[sample_ids]
    else:
        return [arr[idx] for idx in sample_ids]
