import termcolor
from itertools import chain
from typing import List

from .func_utils import ObjectWrapper


def left_aligned(str_list: List[str]) -> List[str]:
    maxlen = max(map(len, str_list), default=0)
    return [f"{s:<{maxlen}}" for s in str_list]


def format_list(lst):
    return ', '.join(map(repr, lst))


def format_path(path: str) -> str:
    return termcolor.colored(path, attrs=['underline'])


def format_id(id_str: str, bracket: bool = True) -> str:
    return termcolor.colored(f"[{id_str}]" if bracket else id_str, 'cyan')


def format_highlight(string: str) -> str:
    bolder = "*" + "-" * (len(string) + 2) + "*"
    return termcolor.colored(
        f"{bolder}\n| {string.upper()} |\n{bolder}",
        color='yellow',
        attrs=['bold'],
    )


def format_highlight2(string: str) -> str:
    return termcolor.colored(string, color='green')


def format_object(obj, *args, **kwargs):
    return f"{obj.__class__.__name__}({join_arg_string(*args, **kwargs)})"


def join_arg_string(*args, sep=', ', **kwargs):
    return sep.join(chain(
        map(str, args),
        (f"{k}={v}" for k, v in kwargs.items()),
    ))


class FormatableMixin:

    def __str__(self):
        return format_object(self, **self.get_config())

    def get_config(self):
        return self.__dict__


class NamedObject(ObjectWrapper):

    def __init__(self, wrapped, name: str):
        super().__init__(wrapped)
        self.name = name

    def __str__(self):
        return self.name


class NamedDict(dict):

    def __setitem__(self, key, val):
        super().__setitem__(key, NamedObject(val, name=key))
