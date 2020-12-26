import json
import os
import pickle
from contextlib import contextmanager
from functools import wraps, lru_cache
from typing import List
from weakref import WeakKeyDictionary

import numpy as np

from .format_utils import format_path
from .func_utils import ObjectWrapper


class cached_property:

    NOT_FOUND = object()

    def __init__(self, func):
        self.__doc__ = func.__doc__
        self.func = func
        self._instance_to_value = WeakKeyDictionary()

    def __get__(self, instance, instance_cls):
        if instance is None:  # just get the cached_property object
            return self

        value = self._instance_to_value.get(instance, self.NOT_FOUND)
        if value is self.NOT_FOUND:
            value = self.func(instance)
            self._instance_to_value[instance] = value

        return value

    def __set__(self, instance, value):
        raise AttributeError("can't set attribute")


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
def reuse_method_call(obj, methods: List[str]):
    wrapped_obj = ObjectWrapper(obj)
    for method_name in methods:
        old_method = getattr(obj, method_name)
        new_method = lru_cache(None)(old_method)
        setattr(wrapped_obj, method_name, new_method)

    yield wrapped_obj

    for method_name in methods:
        getattr(wrapped_obj, method_name).cache_clear()
