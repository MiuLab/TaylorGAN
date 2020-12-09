from argparse import Namespace


class NestedNamespace(Namespace):

    def __setattr__(self, name: str, value):
        names = self._split_first_dot(name)
        if len(names) == 2:
            attr_name, rest = names
            if self.__contains__(attr_name):
                sub_namespace = self.__getattribute__(attr_name)
            else:
                sub_namespace = self.__class__()
                super().__setattr__(attr_name, sub_namespace)
            setattr(sub_namespace, rest, value)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        names = self._split_first_dot(name)
        if len(names) == 2:
            attr_name, rest = names
            return super().__getattribute__(attr_name).__getattr__(rest)
        else:
            return super().__getattribute__(name)

    @staticmethod
    def _split_first_dot(string):
        lst = string.split('.', maxsplit=1)
        if not all(lst):
            raise ValueError(f"invalid name: {string}")
        return lst
