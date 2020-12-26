import os
from typing import Dict

import tensorflow as tf

from library.utils import format_path

from .base import Callback


class ModelCheckpoint(Callback):

    def __init__(self, directory: str, period: int):
        self.directory = directory
        if period <= 0:
            raise ValueError("'saving_period' should be positive!")
        self.period = period

    def on_train_begin(self, logs: Dict = None):
        os.makedirs(self.directory, exist_ok=True)
        if not logs['is_restored']:
            with open(os.path.join(self.directory, "args"), "w") as f:
                f.write(logs['arg_string'])

        self.saver = tf.train.Saver(max_to_keep=2)

    def on_epoch_end(self, epoch: int):
        if epoch % self.period == 0:
            print(f"{epoch} epochs done.")
            path = self.saver.save(
                sess=tf.get_default_session(),
                save_path=os.path.join(self.directory, 'epoch'),
                write_meta_graph=False,
                global_step=epoch,
            )
            print(f"saving checkpoint to {path}.")

    def __str__(self):
        return f"{self.__class__.__name__}(dir={format_path(self.directory)}, period={self.period})"
