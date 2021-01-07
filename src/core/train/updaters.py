import abc

from core.models import Generator, Discriminator
from library.utils import format_id, reuse_method_call, logging_indent

from .optimizer import OptimizerWrapper
from .pubsub_base import Subject


class ModuleUpdater(Subject):

    def __init__(self, module, optimizer: OptimizerWrapper, losses: list):
        self.module = module
        self.optimizer = optimizer
        self.losses = losses

        self.step = 0
        super().__init__()

    @abc.abstractmethod
    def update_step(self):
        pass

    @property
    def info(self):
        return f"{self.module.scope[0]} {format_id(self.module.name)}"

    def summary(self):
        with logging_indent(self.module.scope):
            with logging_indent("Model"):
                print(
                    "Trainable     params:,"
                    f"{count_numel(self.module.trainable_variables):>12}",
                )
                print(
                    "Non-trainable params: "
                    f"{count_numel(self.module.non_trainable_variables):>12,}",
                )

            print(f"Optimizer: {self.optimizer}")
            with logging_indent("Objective:"):
                for loss in self.losses:
                    print(loss)


def count_numel(params) -> int:
    return sum(p.numel() for p in params)


class GeneratorUpdater(ModuleUpdater):

    def update_step(self, real_samples):
        with reuse_method_call(self.generator, ['generate']) as generator:
            loss_collection = sum(
                loss(generator=generator, real_samples=real_samples)
                for loss in self.losses
            )

        # TODO, tensor for checkpoint
        self.step += 1
        self.optimizer.zero_grad()
        loss_collection.total.backward()
        losses = {
            key: tensor.detach().numpy()
            for key, tensor in loss_collection.observables.items()
        }
        for subscriber in self._subscribers:
            subscriber.update(self.step, losses)

        self.optimizer.step()

    @property
    def generator(self) -> Generator:
        return self.module


class DiscriminatorUpdater(ModuleUpdater):

    def update_step(self, real_samples, fake_samples):
        with reuse_method_call(
            self.discriminator,
            ['score_samples', 'score_word_vector', 'get_embedding'],
        ) as discriminator:
            loss_collection = sum(
                loss(
                    discriminator=discriminator,
                    real_samples=real_samples,
                    fake_samples=fake_samples,
                )
                for loss in self.losses
            )

        # TODO, tensor for checkpoint
        self.step += 1
        self.optimizer.zero_grad()
        loss_collection.total.backward()
        losses = {
            key: tensor.detach().numpy()
            for key, tensor in loss_collection.observables.items()
        }
        for subscriber in self._subscribers:
            subscriber.update(self.step, losses)

        self.optimizer.step()

    @property
    def discriminator(self) -> Discriminator:
        return self.module
