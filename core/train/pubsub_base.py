import abc
from typing import List


class Subscriber(abc.ABC):

    @abc.abstractmethod
    def update(self, step, losses):
        pass


class Subject:

    def __init__(self):
        self._subscribers: List[Subscriber] = []

    def attach_subscriber(self, subcriber: Subscriber):
        self._subscribers.append(subcriber)
