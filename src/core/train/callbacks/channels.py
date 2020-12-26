from numbers import Number
from typing import Dict

from core.train.pubsub_base import Subject


class MessageChannel(Subject):

    def post(self, step: int, vals: Dict[str, Number]):
        for subscriber in self._subscribers:
            subscriber.update(step, vals)


channels: Dict[str, MessageChannel] = {}


def register_channel(key: str):
    return channels.setdefault(key, MessageChannel())
