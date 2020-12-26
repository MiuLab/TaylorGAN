import sys
from unittest.mock import patch

from ..logging import logging_indent, TqdmRedirector, PRINT, _IndentPrinter


class TestLoggingIndent:

    def test_recover_builtin_print(self):
        assert print == PRINT  # noqa
        assert _IndentPrinter.level == 0
        with logging_indent():
            # partial
            assert print.func == _IndentPrinter.print_body  # noqa
            assert _IndentPrinter.level == 1
            with logging_indent():
                assert print.func == _IndentPrinter.print_body  # noqa
                assert _IndentPrinter.level == 2
            assert print.func == _IndentPrinter.print_body  # noqa
            assert _IndentPrinter.level == 1

        assert print == PRINT  # noqa
        assert _IndentPrinter.level == 0


class TestTqdmRedirector:

    def test_before_enable(self):
        assert print == TqdmRedirector.PRINT  # noqa

    def test_redirect_ports_after_enable(self):
        TqdmRedirector.enable()
        with patch('tqdm.tqdm.write') as tqdm_write:
            print('a', 'b', 'c')
            tqdm_write.assert_called_with('a b c', file=TqdmRedirector.STDOUT)

    def test_dont_affect_fileIO(self, tmpdir):
        # won't interfere file IO
        filepath = tmpdir / 'output_stream'
        with open(filepath, 'w') as f_out:
            print('a', 'b', 'c', file=f_out)
        with open(filepath, 'r') as f_in:
            assert f_in.readlines() == ['a b c\n']

    def test_after_disable(self):
        TqdmRedirector.disable()
        assert print == TqdmRedirector.PRINT  # noqa
        assert sys.stdout == TqdmRedirector.STDOUT
