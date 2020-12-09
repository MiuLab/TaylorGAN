import abc
from itertools import starmap

from library.utils import allow_abbrev_kwargs, format_id, get_args


class SimpleFactory(abc.ABC):

    def __init__(self, configs=None):
        self.table = {}
        for key, val in (configs or {}).items():
            self.register(key, val)

    def register(self, key: str, func: callable):
        self.table[key] = allow_abbrev_kwargs(func)

    @abc.abstractmethod
    def create(self):
        pass

    @abc.abstractmethod
    def add_argument_to(self, holder):
        pass


def get_help_of_id_kwargs(table):

    def format_item(key, val):
        func = getattr(val, '__wrapped__', val)
        arg_string = ", ".join(get_args(func))
        return f"{format_id(key, bracket=False)}({arg_string})"

    return "Registry & custom options: \n" + "\n".join(starmap(format_item, table.items()))
