import os
import sys
from pathlib import Path

from library.utils import format_path

from core.train.trainers import Trainer

from .base import Callback


class ModelCheckpoint(Callback):

    def __init__(self, trainer: Trainer, directory: Path, period: int):
        self.trainer = trainer
        self.directory = directory
        if period <= 0:
            raise ValueError("'saving_period' should be positive!")
        self.period = period

    def on_train_begin(self, is_restored: bool):
        self.directory.mkdir(exist_ok=True)
        if not is_restored:
            with open(self.directory / "args", "w") as f:
                f.write(" ".join(sys.argv))

    def on_epoch_end(self, epoch: int):
        if epoch % self.period == 0:
            print(f"{epoch} epochs done.")
            path = self.directory / self.checkpoint_basename(epoch)
            self.trainer.save_state(path)
            print(f"saving checkpoint to {format_path(path)}")

    def get_config(self):
        return {'directory': format_path(self.directory), 'period': self.period}

    @staticmethod
    def checkpoint_basename(epoch: int):
        return f'epoch{epoch}.pth'

    @staticmethod
    def epoch_number(path):
        return int(os.path.basename(path)[5:-4])

    @classmethod
    def latest_checkpoint(cls, directory) -> str:
        filename = max(
            (filename for filename in os.listdir(directory) if filename.endswith('.pth')),
            key=cls.epoch_number,
        )
        return os.path.join(directory, filename)
