import abc
from itertools import chain
from operator import attrgetter

from more_itertools import unique_everseen
from tensorflow.python.keras.utils.layer_utils import count_params

from library.utils import logging_block


class ModuleInterface(abc.ABC):

    def __init__(self, name: str = None):
        if not name:
            name = self.__class__.__name__
        self.name = name

    @property
    def trainable_variables(self):
        return self._concat_network_attributes('trainable_variables')

    @property
    def non_trainable_variables(self):
        return self._concat_network_attributes('non_trainable_variables')

    @property
    def updates(self):
        return self._concat_network_attributes('updates')

    def _concat_network_attributes(self, name):
        attrs = chain.from_iterable(map(attrgetter(name), self.networks))
        return list(unique_everseen(attrs))

    @property
    @abc.abstractmethod
    def networks(self):
        raise NotImplementedError

    def summary(self):
        with logging_block(self.scope):
            trainable_params = count_params(self.trainable_variables)
            non_trainable_params = count_params(self.non_trainable_variables)
            print(f"Trainable     params: {trainable_params:>12,}")
            print(f"Non-trainable params: {non_trainable_params:>12,}")
