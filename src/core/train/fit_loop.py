from more_itertools import ilen

from library.utils import batch_generator, format_highlight
from core.preprocess import TextDataset
from .callbacks import NullCallback


class DataLoader:

    def __init__(self, dataset: TextDataset, batch_size: int, n_epochs: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.callback = NullCallback()
        self.batch = 0
        self.epoch = 1

    def skip_epochs(self, epochs):
        # exhaust batch_generator to make sure random state is same
        self.batch += sum(ilen(self._get_batch_generator()) for _ in range(epochs))
        print(f"Skip {epochs} epochs. Finish restoring process.")
        self.epoch += epochs

    def __iter__(self):
        self.callback.on_train_begin(is_restored=self.epoch > 1)
        print(format_highlight("Start Training"))
        while self.epoch <= self.n_epochs:
            self.callback.on_epoch_begin(self.epoch)
            for batch_data in self._get_batch_generator():
                self.callback.on_batch_begin(self.batch)
                yield batch_data
                self.batch += 1
                self.callback.on_batch_end(self.batch, batch_data)

            self.callback.on_epoch_end(self.epoch)
            self.epoch += 1

        self.callback.on_train_end()

    def _get_batch_generator(self):
        return batch_generator(
            self.dataset.ids,
            batch_size=self.batch_size,
            shuffle=True,
            full_batch_only=True,
        )

    @property
    def total(self):
        return len(self.dataset)
