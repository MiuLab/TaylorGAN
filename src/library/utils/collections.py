from functools import reduce
from typing import Any, Dict, Iterable


def counter_or(dicts: Iterable[Dict[Any, int]]):
    return reduce(counter_ior, dicts, {})


def counter_ior(a: Dict[Any, int], b: Dict[Any, int]):
    # NOTE much faster than Counter() |
    for key, cnt in b.items():
        if cnt > a.get(key, 0):
            a[key] = cnt
    return a
