import builtins
import sys
from contextlib import contextmanager
from functools import partial
from typing import List

import termcolor
from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile


SEPARATION_LINE = termcolor.colored(' '.join(['-'] * 50), attrs=['dark'])


def left_aligned(str_list: List[str]) -> List[str]:
    maxlen = max(map(len, str_list), default=0)
    return [f"{s:<{maxlen}}" for s in str_list]


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


def format_object(obj, get_attrs=False, **kwargs):
    if get_attrs:
        kwargs.update(obj.__dict__)
    kwargs_str = ', '.join(f"{k}={v}" for k, v in kwargs.items())
    return f"{obj.__class__.__name__}({kwargs_str})"


@contextmanager
def logging_block(header: str = None, bullet: bool = True):
    '''DO NOT use it with TqdmRedirector!!!'''
    _IndentPrinter.print_header(header)
    _IndentPrinter.indent()

    old_print = builtins.print
    builtins.print = partial(_IndentPrinter.print_body, bullet=bullet)

    yield

    builtins.print = old_print
    _IndentPrinter.dedent()
    _IndentPrinter.print_footer()


class _IndentPrinter:

    '''DO NOT use it with TqdmRedirector!!!'''

    level = 0
    PRINT = builtins.print
    BULLETS = ["•", "–", "*", "·"]

    @classmethod
    def indent(cls):
        if cls.level == 0:  # maybe print is redirected somewhere...
            cls.PRINT = builtins.print
        cls.level += 1

    @classmethod
    def dedent(cls):
        cls.level -= 1

    @classmethod
    def print_header(cls, header):
        if cls.level == 0:
            print(format_highlight(header))
        elif cls.level == 1:
            print(format_highlight2(header))
        else:
            print(header)

    @classmethod
    def print_body(cls, *args, bullet: bool = True, **kwargs):
        if cls.level < 2:
            return cls.PRINT(*args, **kwargs)

        if bullet:
            bullet_symbol = cls.BULLETS[min(cls.level - 1, len(cls.BULLETS) - 1)]
            cls.PRINT(' ' * (2 * cls.level - 2) + bullet_symbol, *args, **kwargs)
        else:
            cls.PRINT(' ' * (2 * cls.level - 3), *args, **kwargs)

    @classmethod
    def print_footer(cls):
        if cls.level == 0:
            print(SEPARATION_LINE)
        elif cls.level == 1:
            print()


class TqdmRedirector:

    # ports before enable() is called
    STDOUT, STDERR, PRINT = sys.stdout, sys.stderr, builtins.print

    @classmethod
    def enable(cls):
        '''DO NOT use it under logging_block!!!'''
        # ports before being redirected, maybe there're redirected somewhere...
        cls.STDOUT, cls.STDERR, cls.PRINT = sys.stdout, sys.stderr, builtins.print
        TQDMOUT, TQDMERR = DummyTqdmFile(sys.stdout), DummyTqdmFile(sys.stderr)
        STREAMS_TO_REDIRECT = {None, cls.STDOUT, cls.STDERR, TQDMOUT, TQDMERR}

        def safe_print(*values, sep=' ', end='\n', file=None, flush=False):
            if file in STREAMS_TO_REDIRECT:
                # NOTE tqdm (v4.40.0) can't support end != '\n' and flush
                tqdm.write(sep.join(map(str, values)), file=cls.STDOUT)
            else:
                cls.PRINT(*values, sep=sep, end=end, file=file, flush=flush)

        sys.stdout, sys.stderr, builtins.print = TQDMOUT, TQDMERR, safe_print

    @classmethod
    def disable(cls):
        sys.stdout, sys.stderr, builtins.print = cls.STDOUT, cls.STDERR, cls.PRINT
