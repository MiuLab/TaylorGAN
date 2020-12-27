import abc
from itertools import chain


class ModuleInterface:

    def __init__(self, name: str = None):
        super().__init__()
        if not name:
            name = self.__class__.__name__
        self.name = name

    def parameters(self):
        return chain.from_iterable(network.parameters() for network in self.networks)

    @property
    def trainable_variables(self):
        return [param for param in self.parameters() if param.require_grad]

    @property
    @abc.abstractmethod
    def networks(self):
        raise NotImplementedError

    @property
    def trainable_params(self):
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    @property
    def non_trainable_params(self):
        return sum(param.numel() for param in self.parameters() if not param.requires_grad)
