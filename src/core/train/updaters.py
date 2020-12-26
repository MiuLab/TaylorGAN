import abc

import tensorflow as tf

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
    def build_graph(self):
        pass

    def _build_train_op(self, loss):
        self._step_tensor = tf.Variable(0, f"{self.module.scope}_step")
        with tf.control_dependencies(self.module.updates):
            self._train_op = self.optimizer.minimize(
                loss,
                var_list=self.module.trainable_variables,
                global_step=self._step_tensor,
            )

    def update_step(self, feed_dict=None):
        _, self.step, losses = tf.get_default_session().run(
            [self._train_op, self._step_tensor, self._observables],
            feed_dict=feed_dict,
        )
        for subscriber in self._subscribers:
            subscriber.update(self.step, losses)

    @property
    def info(self):
        return f"{self.module.scope[0]} {format_id(self.module.name)}"

    def summary(self):
        with logging_indent(self.module.scope):
            with logging_indent("Model"):
                print(f"Trainable     params: {self.module.trainable_params:>12,}")
                print(f"Non-trainable params: {self.module.non_trainable_params:>12,}")

            self.optimizer.summary()
            with logging_indent("Objective:"):
                for loss in self.losses:
                    print(loss)


class GeneratorUpdater(ModuleUpdater):

    def build_graph(self, real_samples):
        with reuse_method_call(
            self.generator,
            ['generate', 'teacher_forcing_generate'],
        ) as generator:
            loss_collection = sum(
                loss(generator=generator, real_samples=real_samples)
                for loss in self.losses
            )

        self._build_train_op(loss_collection.total)
        self._observables = loss_collection.observables

    @property
    def generator(self) -> Generator:
        return self.module


class DiscriminatorUpdater(ModuleUpdater):

    def build_graph(self, real_samples, fake_samples):
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

        self._build_train_op(loss_collection.total)
        self._observables = loss_collection.observables

    @property
    def discriminator(self) -> Discriminator:
        return self.module
