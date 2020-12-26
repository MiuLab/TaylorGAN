import builtins
import sys
import warnings
from contextlib import contextmanager
from functools import partial

import termcolor
from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile

from .format_utils import format_highlight, format_highlight2


STDOUT, STDERR, PRINT = sys.stdout, sys.stderr, builtins.print  # guaranteed builtins!!
SEPARATION_LINE = termcolor.colored(' '.join(['-'] * 50), attrs=['dark'])


@contextmanager
def logging_indent(header: str = None, bullet: bool = True):
    if header:
        _IndentPrinter.print_header(header)

    if _IndentPrinter.level == 0:  # need redirect
        if builtins.print != PRINT:
            warnings.warn("`logging_indent` should not be used with other redirector!")
        builtins.print = partial(_IndentPrinter.print_body, bullet=bullet)

    _IndentPrinter.level += 1
    yield
    _IndentPrinter.level -= 1

    if _IndentPrinter.level == 0:  # need recover
        builtins.print = PRINT
    _IndentPrinter.print_footer()


class _IndentPrinter:

    '''DO NOT use it with TqdmRedirector!!!'''

    BULLETS = ["•", "–", "*", "·"]
    level = 0

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
        assert cls.level > 0
        if cls.level < 2:
            PRINT(*args, **kwargs)
        elif bullet:
            bullet_symbol = cls.BULLETS[min(cls.level, len(cls.BULLETS)) - 1]
            PRINT(' ' * (2 * cls.level - 2) + bullet_symbol, *args, **kwargs)
        else:
            PRINT(' ' * (2 * cls.level - 3), *args, **kwargs)

    @classmethod
    def print_footer(cls):
        if cls.level == 0:
            print(SEPARATION_LINE)
        elif cls.level == 1:
            print()


class TqdmRedirector:

    STDOUT, STDERR, PRINT = STDOUT, STDERR, PRINT

    @classmethod
    def enable(cls):
        if (sys.stdout, sys.stderr, builtins.print) != (STDOUT, STDERR, PRINT):
            warnings.warn(f"`{cls.__name__}` should not be used with other redirector!")

        tqdm_out, tqdm_err = DummyTqdmFile(STDOUT), DummyTqdmFile(STDERR)
        STREAMS_TO_REDIRECT = {None, STDOUT, STDERR, tqdm_out, tqdm_err}

        def new_print(*values, sep=' ', end='\n', file=None, flush=False):
            if file in STREAMS_TO_REDIRECT:
                # NOTE tqdm (v4.40.0) can't support end != '\n' and flush
                tqdm.write(sep.join(map(str, values)), file=STDOUT)
            else:
                PRINT(*values, sep=sep, end=end, file=file, flush=flush)

        sys.stdout, sys.stderr, builtins.print = tqdm_out, tqdm_err, new_print

    @classmethod
    def disable(cls):
        sys.stdout, sys.stderr, builtins.print = STDOUT, STDERR, PRINT
