import inspect

from ..func_utils import ObjectWrapper, wraps_with_new_signature


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


def test_wraps_with_new_signature():

    def foo(a, b):
        pass

    @wraps_with_new_signature(foo)
    def wrapper(*args, c, **kwargs):
        pass

    assert list(inspect.signature(wrapper).parameters.keys()) == ['a', 'b', 'c']
