import pytest

from ..func_utils import log_args_when_error, match_abbrev, ObjectWrapper


def test_log_args_when_error():

    @log_args_when_error
    def foo(x, y, z):
        pass

    with pytest.raises(TypeError) as e:
        foo(1, 2)
        assert e.msg == "allowed arguments of foo: 'x', 'y', 'z'"


def test_abbrev_kwargs():

    @match_abbrev
    def format_fruit(apple=1, banana=2, cherry=3, bamboo=None):
        return f"A={apple}, B={banana}, C={cherry}"

    assert format_fruit(che=3) == "A=1, B=2, C=3"
    assert format_fruit(cher=4, bana=5) == "A=1, B=5, C=4"

    with pytest.raises(TypeError):
        format_fruit(1, app=1)  # raise by original function
    with pytest.raises(TypeError):
        format_fruit(appleeee=1)  # doesn't match
    with pytest.raises(TypeError):
        format_fruit(ba=1)  # 1 abbrev -> 2 keywords
    with pytest.raises(TypeError):
        format_fruit(app=1, appl=2)  # 2 abbrevs -> 1 keyword


def test_object_wrapper():
    class A:

        def foo(self):
            return 'foo'

    class B(ObjectWrapper):

        def goo(self):
            return 'goo'

    a = A()
    b = B(a)
    assert b.foo() == a.foo()
    assert b.goo() == 'goo'
