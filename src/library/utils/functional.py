import inspect
from functools import wraps
from typing import List


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


def allow_abbrev_kwargs(func):
    func_args = get_args(func)

    @wraps(func)
    def wrapped(*args, **kwargs):
        new_kwargs = {}
        for abbrev, val in kwargs.items():
            matched = [kw for kw in func_args if kw.startswith(abbrev)]
            if len(matched) == 1:
                target_kwarg = matched[0]
                if target_kwarg in new_kwargs:
                    raise TypeError(f"multiple abbreviations match {target_kwarg}")
                new_kwargs[target_kwarg] = val
            elif len(matched) > 1:
                raise TypeError(
                    f"ambiguous abbreviation: {repr(abbrev)} could match {format_list(matched)}",
                )
            # else len = 0
            elif inspect.getfullargspec(func).varkw is not None:
                new_kwargs[abbrev] = val
            else:
                raise TypeError(
                    f"{func.__qualname__} got an unexpected keyword argument {repr(abbrev)}, "
                    f"allowed arguments of {func.__qualname__}: {format_list(func_args)}",
                )

        return func(*args, **new_kwargs)

    return wrapped


def get_args(func) -> List[str]:
    func_args = inspect.getfullargspec(func).args
    if func_args and func_args[0] in ('self', 'cls'):
        return func_args[1:]
    return func_args


def format_list(lst):
    return ', '.join(map(repr, lst))


class ObjectWrapper:

    def __init__(self, body):
        self._body = body

    def __getattr__(self, name):
        return getattr(self._body, name)
