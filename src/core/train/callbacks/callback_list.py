from itertools import chain
from typing import Dict, Sequence

from library.utils import logging_block

from .base import Callback


class CallbackList(Callback):

    def __init__(
            self,
            evaluater: Callback = (),
            loggers: Sequence[Callback] = (),
            others: Sequence[Callback] = (),
        ):
        self.evaluater = evaluater
        self.loggers = list(loggers)
        self.others = list(others)

    def on_train_begin(self, logs: Dict = None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_epoch_begin(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)

    def on_batch_begin(self, batch: int):
        for callback in self.callbacks:
            callback.on_batch_begin(batch)

    def on_batch_end(self, batch: int, batch_data):
        for callback in self.callbacks:
            callback.on_batch_end(batch, batch_data)

    def on_epoch_end(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch)

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()

    def summary(self):
        with logging_block("Callbacks:"):
            for cbk in chain(self.loggers, self.others):
                print(cbk)

    @property
    def callbacks(self):
        return chain([self.evaluater], self.loggers, self.others)
