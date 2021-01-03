import abc
from itertools import chain


class ModuleInterface:

    def parameters(self):
        return chain.from_iterable(network.parameters() for network in self.networks)

    def modules(self):
        return chain.from_iterable(network.modules() for network in self.networks)

    @property
    def trainable_variables(self):
        return [param for param in self.parameters() if param.requires_grad]

    @property
    def non_trainable_variables(self):
        return [param for param in self.parameters() if not param.requires_grad]

    @property
    @abc.abstractmethod
    def networks(self):
        raise NotImplementedError
