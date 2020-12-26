import abc

from library.utils import logging_block

from .updaters import ModuleUpdater


class Trainer(abc.ABC):

    def __init__(self, data_source, generator_updater: ModuleUpdater):
        self.data_source = data_source
        self.generator_updater = generator_updater

    @abc.abstractmethod
    def fit_batch(self, batch_data):
        pass

    @property
    def updaters(self):
        return [self.generator_updater]

    def summary(self):
        with logging_block('Model Summary'):
            for updater in self.updaters:
                updater.module.summary()


class GANTrainer(Trainer):

    def __init__(
            self,
            data_source,
            generator_updater: ModuleUpdater,
            objective_updater: ModuleUpdater,
            d_steps: int,
        ):
        super().__init__(data_source, generator_updater)
        self.objective_updater = objective_updater
        self.d_steps = d_steps

    def fit_batch(self, batch_data):
        self.objective_updater.update_step(self.data_source.get_feed_dict(batch_data))
        if self.objective_updater.step % self.d_steps == 0:
            self.generator_updater.update_step()

    @property
    def updaters(self):
        return super().updaters + [self.objective_updater]
