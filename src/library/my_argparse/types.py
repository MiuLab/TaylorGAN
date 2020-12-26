import math
import os


def int_in_range(minval=-math.inf, maxval=math.inf):

    def func(x):
        return _validate_in_range(int(x), minval, maxval, inclusive=True)

    if (minval, maxval) == (1, math.inf):
        func.__name__ = 'positive_int'
    elif (minval, maxval) == (0, math.inf):
        func.__name__ = 'nonnegative_int'
    else:
        func.__name__ = f'int∈{_repr_inteval(minval, maxval, inclusive=True)}'
    return func


def float_in_range(minval=-math.inf, maxval=math.inf, inclusive=True):

    def func(x):
        return _validate_in_range(float(x), minval, maxval, inclusive=inclusive)

    if (minval, maxval, inclusive) == (0., math.inf, False):
        func.__name__ = 'positive_float'
    elif (minval, maxval, inclusive) == (0., math.inf, True):
        func.__name__ = 'nonnegative_float'
    else:
        func.__name__ = f'float∈{_repr_inteval(minval, maxval, inclusive)}'
    return func


def _validate_in_range(x, minval, maxval, inclusive):
    if math.isinf(x):
        raise ValueError
    if inclusive:
        if not (minval <= x <= maxval):
            raise ValueError
    elif not (minval < x < maxval):
        raise ValueError
    return x


def _repr_inteval(minval, maxval, inclusive):

    def _math_repr(x):
        if x == -math.inf:
            return '-∞'
        if x == math.inf:
            return '∞'
        return repr(x)

    left_bracket = '[' if (inclusive and not math.isinf(minval)) else '('
    right_bracket = ']' if (inclusive and not math.isinf(maxval)) else ')'
    return f"{left_bracket}{_math_repr(minval)}, {_math_repr(maxval)}{right_bracket}"


def path(x):
    return os.path.normpath(x)


def filepath(x):
    """Check if file exists.
    """
    x = path(x)
    if not os.path.isfile(x):
        raise ValueError
    return x


def dirpath(x):
    """Check if directory exists.
    """
    x = path(x)
    if not os.path.isdir(x):
        raise ValueError
    return x
