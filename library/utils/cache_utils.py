import json
import os
import pickle
from contextlib import contextmanager
from functools import wraps, lru_cache
from typing import List

import numpy as np

from .logging import format_path


_NOT_FOUND = object()


class cached_property:

    # Simplified https://github.com/python/cpython/blob/3.8/Lib/functools.py#L928-L976

    def __init__(self, func):
        self.func = func
        self.attrname = None
        self.__doc__ = func.__doc__

    def __set_name__(self, owner, name):
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise TypeError(
                "Cannot assign the same cached_property to two different names "
                f"({self.attrname!r} and {name!r}).",
            )

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if self.attrname is None:
            raise TypeError(
                "Cannot use cached_property instance without calling __set_name__ on it.",
            )

        cache = instance.__dict__
        val = cache.get(self.attrname, _NOT_FOUND)
        if val is _NOT_FOUND:
            val = self.func(instance)
            cache[self.attrname] = val
        return val


class FileCache:

    @classmethod
    def tofile(cls, path, makedirs: bool = True):
        def decorator(func):
            @wraps(func)
            def wrapped(*args, **kwargs):
                path_str = cls._get_path_str(path, *args, **kwargs)
                if os.path.isfile(path_str):
                    print(f"Load from {format_path(path_str)}")
                    return cls.load_data(path_str)

                output = func(*args, **kwargs)
                if makedirs:
                    os.makedirs(os.path.dirname(path_str), exist_ok=True)

                print(f"Cache to {format_path(path_str)}")
                cls.save_data(output, path_str)
                return output

            return wrapped

        return decorator

    @staticmethod
    def _get_path_str(path, *args, **kwargs):
        if callable(path):
            return path(*args, **kwargs)
        else:
            return str(path).format(*args, **kwargs)

    def load_data(path):
        raise NotImplementedError

    def save_data(data, path):
        raise NotImplementedError


class PickleCache(FileCache):

    @staticmethod
    def load_data(path):
        with open(path, 'rb') as f_in:
            return pickle.load(f_in)

    @staticmethod
    def save_data(data, path):
        with open(path, 'wb') as f_out:
            pickle.dump(data, f_out)


class NumpyCache(FileCache):

    @staticmethod
    def load_data(path):
        return np.load(path)['data']

    @staticmethod
    def save_data(data, path):
        np.savez_compressed(path, data=data)


_subclass_map = {}


class JSONSerializableMixin:

    @classmethod
    def __init_subclass__(cls):
        _subclass_map[cls.__name__] = cls

    def serialize(self):
        return json.dumps(
            {
                'class_name': self.__class__.__name__,
                'config': self.get_config(),
            },
            indent=2,
        )

    @classmethod
    def deserialize(cls, data: str):
        params = json.loads(data)
        subclass = _subclass_map[params['class_name']]
        if not issubclass(subclass, cls):
            raise ValueError(
                f"{cls.__name__}.deserialize on non-subclass {subclass.__name__} is forbidden!",
            )
        return subclass.from_config(params['config'])

    def save(self, path):
        with open(path, 'w') as f_out:
            f_out.write(self.serialize())

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f_in:
            return cls.deserialize(f_in.read())

    def get_config(self):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config_dict):
        # NOTE is not necessary to be dict,
        #      but should be consistent with return type of get_config
        return cls(**config_dict)

    def __eq__(self, other):
        return (self.__class__, self.get_config()) == (other.__class__, other.get_config())


class JSONCache(FileCache):

    @staticmethod
    def load_data(path):
        return JSONSerializableMixin.load(path)

    @staticmethod
    def save_data(instance, path: str):
        instance.save(path)


@contextmanager
def cache_method_call(obj, methods: List[str]):
    old_methods = [getattr(obj, name) for name in methods]
    new_methods = [lru_cache(None)(om) for om in old_methods]
    for name, new_method in zip(methods, new_methods):
        setattr(obj, name, new_method)

    yield

    for new_method in new_methods:
        new_method.cache_clear()
    for name, old_method in zip(methods, old_methods):
        setattr(obj, name, old_method)
