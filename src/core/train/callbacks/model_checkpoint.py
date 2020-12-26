import sys
from pathlib import Path

import tensorflow as tf

from library.utils import format_path

from .base import Callback


class ModelCheckpoint(Callback):

    def __init__(self, directory: Path, period: int):
        self.directory = directory
        if period <= 0:
            raise ValueError("'saving_period' should be positive!")
        self.period = period

    def on_train_begin(self, is_restored: bool):
        self.directory.mkdir(exist_ok=True)
        if not is_restored:
            with open(self.directory / "args", "w") as f:
                f.write(" ".join(sys.argv))

        self.saver = tf.train.Saver(max_to_keep=2)

    def on_epoch_end(self, epoch: int):
        if epoch % self.period == 0:
            print(f"{epoch} epochs done.")
            path = self.saver.save(
                sess=tf.get_default_session(),
                save_path=str(self.directory / 'epoch'),
                write_meta_graph=False,
                global_step=epoch,
            )
            print(f"saving checkpoint to {path}.")

    def get_config(self):
        return {'directory': format_path(self.directory), 'period': self.period}
