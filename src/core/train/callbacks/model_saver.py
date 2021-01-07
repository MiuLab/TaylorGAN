from pathlib import Path

import torch

from library.utils import format_path

from .base import Callback


class ModelSaver(Callback):

    def __init__(self, module, directory: Path, period: int):
        self.module = module
        self.directory = directory
        if period <= 0:
            raise ValueError("'saving_period' should be positive!")
        self.period = period

    def on_epoch_end(self, epoch):
        if epoch % self.period == 0:
            path = self.directory / f"model_epo{epoch}.pth"
            print(f"{epoch} epochs done. Save model to {format_path(path)}.")
            traced = self.module.export_traced()
            torch.jit.save(traced, str(path))

    def __str__(self):
        return f"{self.__class__.__name__}(dir={format_path(self.directory)}, period={self.period})"
