import argparse
import pytest
from unittest.mock import patch

from ..parser import MyArgumentParser


@pytest.mark.parametrize('arg', [
    'main.py -h',
    'main.py --help',
])
def test_help_formatter(arg):
    parser = MyArgumentParser()

    parser.add_argument('x')
    parser.add_argument('--y', help=argparse.SUPPRESS)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--foo', choices=['a', 'b', 'c'], action='append')
    group.add_argument('--boo', choices=['a', 'b', 'c'], action='append')

    with patch('sys.argv', arg.split()), pytest.raises(SystemExit) as exc_info:
        parser.parse_args()
    assert exc_info.value.code == 0
