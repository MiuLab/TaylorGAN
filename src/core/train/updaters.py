import tensorflow as tf

from library.utils import format_id, cache_method_call
from core.objectives.collections import LossCollection

from .pubsub_base import Subject


class ModuleUpdater(Subject):

    def __init__(self, module, optimizer):
        self.module = module
        self.optimizer = optimizer

        self.step = 0
        self._losses = []
        super().__init__()

    def add_loss(self, loss: LossCollection):
        self._losses.append(loss)

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


class GeneratorUpdater(ModuleUpdater):

    def build_graph(self, real_samples):
        with cache_method_call(self.generator, ['generate', 'teacher_forcing_generate']):
            loss_collection = sum(
                loss(generator=self.generator, real_samples=real_samples)
                for loss in self._losses
            )

        self._build_train_op(loss_collection.total)
        self._observables = loss_collection.observables

    @property
    def generator(self):
        return self.module


class DiscriminatorUpdater(ModuleUpdater):

    def build_graph(self, real_samples, fake_samples):
        with cache_method_call(self.discriminator, ['score_samples', 'score_word_vector']):
            loss_collection = sum(
                loss(
                    discriminator=self.discriminator,
                    real_samples=real_samples,
                    fake_samples=fake_samples,
                )
                for loss in self._losses
            )

        self._build_train_op(loss_collection.total)
        self._observables = loss_collection.observables

    @property
    def discriminator(self):
        return self.module
