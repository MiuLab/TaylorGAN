from typing import Dict


class Callback:

    def on_train_begin(self, logs: Dict = None):
        pass

    def on_epoch_begin(self, epoch: int):
        pass

    def on_batch_begin(self, batch: int):
        pass

    def on_batch_end(self, batch: int, batch_data):
        pass

    def on_epoch_end(self, epoch: int):
        pass

    def on_train_end(self):
        pass


NullCallback = Callback
