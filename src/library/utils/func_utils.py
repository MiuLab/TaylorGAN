import inspect
from functools import WRAPPER_ASSIGNMENTS, WRAPPER_UPDATES, update_wrapper
from itertools import chain


class ObjectWrapper:

    def __init__(self, wrapped):
        self._wrapped = wrapped

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


class ArgumentBinder:

    def __init__(self, func, preserved=()):
        old_sig = inspect.signature(func)
        preserved = set(preserved)
        self.func = func
        self.__signature__ = old_sig.replace(
            parameters=[
                param
                for key, param in old_sig.parameters.items()
                if param.name not in preserved
            ],
        )

    def __call__(self, *b_args, **b_kwargs):
        binding = self.__signature__.bind_partial(*b_args, **b_kwargs)

        def bound_function(*args, **kwargs):
            return self.func(
                *args,
                **kwargs,
                **binding.arguments,
            )

        return bound_function


def wraps_with_new_signature(wrapped, assigned=WRAPPER_ASSIGNMENTS, updated=WRAPPER_UPDATES):

    def update_wrapper_signature(wrapper):
        wrapper = update_wrapper(wrapper, wrapped=wrapped, assigned=assigned, updated=updated)
        old_sig = inspect.signature(wrapped)
        add_params = [
            p
            for p in inspect.signature(wrapper, follow_wrapped=False).parameters.values()
            if p.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
        ]
        joined_parameters = sorted(
            chain(old_sig.parameters.values(), add_params),
            key=lambda p: (p.kind, p.default != p.empty),  # empty first
        )
        new_sig = old_sig.replace(parameters=joined_parameters)
        wrapper.__signature__ = new_sig
        return wrapper

    return update_wrapper_signature
