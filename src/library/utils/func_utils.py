import inspect
from functools import wraps
from typing import List

from .collections import dict_of_unique
from .format_utils import format_list


def log_args_when_error(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        try:
            output = func(*args, **kwargs)
        except TypeError:
            func_args = get_args(func)
            raise TypeError(f"allowed arguments of {func.__qualname__}: {format_list(func_args)}")
        return output

    return wrapped


def match_abbrev(func):
    func_args = get_args(func)
    bypass = inspect.getfullargspec(func).varkw is not None

    def match_abbrev(abbrev):
        matches = [kw for kw in func_args if kw.startswith(abbrev)]
        if len(matches) > 1:
            raise TypeError(
                f"ambiguous: {abbrev} match multiple results: {format_list(matches)}",
            )
        if len(matches) == 1:
            return matches[0]
        elif bypass:  # too short
            return abbrev

        raise TypeError(
            f"{func.__qualname__} got an unexpected keyword argument {repr(abbrev)}, "
            f"allowed arguments of {func.__qualname__}: {format_list(func_args)}",
        )

    @wraps(func)
    def wrapped(*args, **kwargs):
        try:
            new_kwargs = dict_of_unique(
                (match_abbrev(key), val) for key, val in kwargs.items()
            )
        except ValueError as e:
            raise TypeError(f"more than one abbrev match to the same keyword: {e}")

        return func(*args, **new_kwargs)

    return wrapped


def get_args(func) -> List[str]:
    func_args = inspect.getfullargspec(func).args
    if func_args and func_args[0] in ('self', 'cls'):
        return func_args[1:]
    return func_args


def extract_wrapped(func, attr_name='__wrapped__'):
    if hasattr(func, attr_name):
        return extract_wrapped(getattr(func, attr_name))
    return func


class ObjectWrapper:

    def __init__(self, body):
        self._body = body

    def __getattr__(self, name):
        return getattr(self._body, name)
