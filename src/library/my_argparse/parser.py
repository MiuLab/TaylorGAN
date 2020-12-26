import argparse

from .formatters import MyFormatter
from .namespace import NestedNamespace


class MyArgumentParser(argparse.ArgumentParser):

    def __init__(
            self,
            prog=None,
            usage=None,
            description=None,
            epilog=None,
            parents=(),
            formatter_class=MyFormatter,
            prefix_chars='-',
            fromfile_prefix_chars=None,
            argument_default=None,
            conflict_handler='error',
            add_help=True,
            allow_abbrev=True,
        ):
        super().__init__(
            prog=prog,
            usage=usage,
            description=description,
            epilog=epilog,
            parents=parents,
            formatter_class=formatter_class,
            prefix_chars=prefix_chars,
            fromfile_prefix_chars=fromfile_prefix_chars,
            argument_default=argument_default,
            conflict_handler=conflict_handler,
            add_help=add_help,
            allow_abbrev=allow_abbrev,
        )

    def parse_known_args(self, args=None, namespace=None):
        if namespace is None:
            namespace = NestedNamespace()
        return super().parse_known_args(args, namespace)
