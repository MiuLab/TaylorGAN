import pytest
from unittest.mock import patch

from ..actions import IdKwargs
from ..parser import MyArgumentParser


class TestStoreIdKwargs:

    @pytest.fixture(scope='class')
    def yoctol_parser(self):
        parser = MyArgumentParser(prog='main.py')
        parser.add_argument(
            '--foo',
            action=IdKwargs,
            id_choices=['a', 'b'],
            use_bool_abbreviation=True,
            sub_action='store',
        )
        return parser

    @pytest.mark.parametrize('arg_string, expected_output', [
        ['main.py --foo a', ('a', {})],
        [
            'main.py --foo a I=1,F1=1.,F2=1e-4,F3=-1e-4',
            ('a', {'I': 1, 'F1': 1., 'F2': 1e-4, 'F3': -1e-4}),
        ],
        [
            'main.py --foo a B1=False,B2=True,B3,N=None',
            ('a', {'B1': False, 'B2': True, 'B3': True, 'N': None}),
        ],
        [
            'main.py --foo a S1=s,S2="s",S3=\'s\'',
            ('a', {'S1': 's', 'S2': 's', 'S3': 's'}),
        ],
        [
            'main.py --foo a 1=open,2=exit,3=exec,4=import,5=OSError',  # no builtins allowed
            ('a', {'1': 'open', '2': 'exit', '3': 'exec', '4': 'import', '5': 'OSError'}),
        ],
    ])
    def test_store(self, yoctol_parser, arg_string, expected_output):
        with patch('sys.argv', arg_string.split(' ')):
            args = yoctol_parser.parse_args()
        assert args.foo == expected_output

    @pytest.mark.parametrize('invalid_arg', [
        pytest.param('main.py --foo', id='nargs<1'),
        pytest.param('main.py --foo a 1 b', id='nargs>2'),
        pytest.param('main.py --foo c 1', id='invalid_choice'),
        pytest.param('main.py --foo a x=1=y', id='invalid_format_='),
        pytest.param('main.py --foo a x=1+y=y', id='invalid_format_split'),
        pytest.param('main.py --foo a x=1,x=2', id='duplicated_key'),
    ])
    def test_raise_invalid_arg(self, yoctol_parser, invalid_arg):
        argv = invalid_arg.split()
        with patch('sys.argv', argv), pytest.raises(SystemExit) as exc_info:
            yoctol_parser.parse_args()
        assert exc_info.value.code == 2
