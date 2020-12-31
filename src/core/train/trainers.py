import abc
from typing import Iterator

import numpy as np
import torch

from core.models.sequence_modeling import TokenSequence
from .updaters import GeneratorUpdater, DiscriminatorUpdater


class Trainer(abc.ABC):

    def __init__(self, generator_updater: GeneratorUpdater):
        self.generator_updater = generator_updater

    @abc.abstractmethod
    def fit(self, data_loader: Iterator[np.ndarray]):
        pass

    @property
    def updaters(self):
        return [self.generator_updater]

    def summary(self):
        for updater in self.updaters:
            updater.summary()


class NonParametrizedTrainer(Trainer):

    def fit(self, data_loader: Iterator[np.ndarray]):
        for batch_data in data_loader:
            real_samples = TokenSequence(
                torch.from_numpy(batch_data).type(torch.long),
                eos_idx=1,
            )
            self.generator_updater.update_step(real_samples)


class GANTrainer(Trainer):

    def __init__(
            self,
            generator_updater: GeneratorUpdater,
            discriminator_updater: DiscriminatorUpdater,
            d_steps: int = 1,
        ):
        super().__init__(generator_updater)
        self.discriminator_updater = discriminator_updater
        self.d_steps = d_steps

    def fit(self, data_loader: Iterator[np.ndarray]):
        for batch_data in data_loader:
            real_samples = TokenSequence(
                torch.from_numpy(batch_data).type(torch.long),
                eos_idx=1,
            )
            self.discriminator_updater.update_step(
                real_samples=real_samples,
                fake_samples=self.generator_updater.generator.generate(*batch_data.shape),
            )
            if self.discriminator_updater.step % self.d_steps == 0:
                self.generator_updater.update_step(real_samples)

    @property
    def updaters(self):
        return super().updaters + [self.discriminator_updater]
