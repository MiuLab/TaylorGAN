import numpy as np
import pytest

from ..iterator import tqdm_open, batch_generator


def test_tqdm_open(tmpdir):
    filepath = tmpdir / 'test_tqdm_open.txt'
    filepath.write('tqdm_open\n' * 100)
    with tqdm_open(filepath, 'r') as f_in:
        assert all(line == 'tqdm_open\n' for line in f_in)


class TestBatchGenerator:

    @pytest.mark.parametrize('full_batch_only', [True, False])
    def test_not_shuffle(self, full_batch_only):
        datum = np.random.choice(100, size=[10, 2])
        batch_size = 3
        batches = list(
            batch_generator(datum, batch_size, shuffle=False, full_batch_only=full_batch_only),
        )
        if full_batch_only:
            assert len(batches) == len(datum) // batch_size
        else:
            assert len(batches) == (len(datum) + batch_size - 1) // batch_size

        for start, batch in zip(range(0, len(datum), batch_size), batches):
            assert (batch == datum[start: start + batch_size]).all()

    def test_shuffle(self):
        data = np.random.choice(100, [10, 3])
        data[:, 0] = np.arange(len(data), dtype=np.int32)  # NOTE memorize index before shuffle

        shuffled_data = batch_generator(
            data,
            3,
            shuffle=True,
            full_batch_only=False,
        )
        shuffled_data = np.concatenate(list(shuffled_data), axis=0)

        shuffled_row_ids = shuffled_data[:, 0]
        np.testing.assert_array_equal(shuffled_data, data[shuffled_row_ids])
