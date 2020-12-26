import abc
from more_itertools import ichunked
from typing import Iterator

import numpy as np

from .updaters import GeneratorUpdater, DiscriminatorUpdater


class Trainer(abc.ABC):

    def __init__(self, placeholder, generator_updater: GeneratorUpdater):
        self.placeholder = placeholder
        self.generator_updater = generator_updater

    @abc.abstractmethod
    def fit(self, data_loader: Iterator[np.ndarray]):
        pass

    @property
    def updaters(self):
        return [self.generator_updater]

    def build_graph(self, real_samples):
        self.generator_updater.build_graph(real_samples)

    def summary(self):
        for updater in self.updaters:
            updater.summary()


class NonParametrizedTrainer(Trainer):

    def fit(self, data_loader: Iterator[np.ndarray]):
        for batch_data in data_loader:
            self.generator_updater.update_step(
                feed_dict={self.placeholder: batch_data},
            )


class GANTrainer(Trainer):

    def __init__(
            self,
            placeholder,
            generator_updater: GeneratorUpdater,
            discriminator_updater: DiscriminatorUpdater,
            d_steps: int = 1,
        ):
        super().__init__(placeholder, generator_updater)
        self.discriminator_updater = discriminator_updater
        self.d_steps = d_steps

    def fit(self, data_loader: Iterator[np.ndarray]):
        for chunk in ichunked(data_loader, n=self.d_steps):
            for batch_data in chunk:
                self.discriminator_updater.update_step(
                    feed_dict={self.placeholder: batch_data},
                )
            self.generator_updater.update_step()

    def build_graph(self, real_samples):
        super().build_graph(real_samples)
        fake_samples = self.generator_updater.generator.generate(
            batch_size=real_samples.batch_size,
            maxlen=real_samples.maxlen,
        )
        self.discriminator_updater.build_graph(
            real_samples=real_samples,
            fake_samples=fake_samples,
        )

    @property
    def updaters(self):
        return super().updaters + [self.discriminator_updater]
