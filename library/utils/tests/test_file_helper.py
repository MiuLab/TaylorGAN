from ..file_helper import count_lines


def test_count_lines(tmpdir):
    filepath = tmpdir / 'test_count_lines.txt'
    filepath.write('\n' * 100)
    assert count_lines(filepath) == 100
