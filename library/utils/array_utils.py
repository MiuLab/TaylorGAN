import numpy as np


def safe_divide(a, b):
    return a / np.maximum(b, 1)


def get_seqlens(data: np.ndarray, eos_idx):
    data = np.asarray(data, dtype=np.int)
    end_mask = np.equal(data, eos_idx)
    return np.where(
        np.any(end_mask, axis=1),
        np.argmax(end_mask, axis=1),  # position of eos
        data.shape[1],  # pad length
    )


def unpad(sequences, lengths):
    return map(lambda s, length: s[:length], sequences, lengths)
