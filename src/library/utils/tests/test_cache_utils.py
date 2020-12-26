import sys
from unittest.mock import Mock

import numpy as np
import pytest

from ..cache_utils import (
    cached_property,
    PickleCache,
    NumpyCache,
    JSONSerializableMixin,
    JSONCache,
    reuse_method_call,
)


class TestCacheProperty:

    def test_does_not_execute_twice(self):
        side_effect = Mock()

        class A:

            @cached_property
            def foo(self):
                side_effect()
                return 'foo'

        a = A()
        assert a.foo == 'foo'
        assert side_effect.call_count == 1
        assert a.foo == 'foo'
        assert side_effect.call_count == 1  # hit

        a2 = A()
        assert a2.foo == 'foo'
        assert side_effect.call_count == 2

    def test_readonly(self):
        class A:

            @cached_property
            def foo(self):
                return 'foo'

        a = A()
        assert a.foo == 'foo'
        with pytest.raises(AttributeError):
            a.foo = 'goo'

    def test_does_not_leak_memory(self):

        class A:

            @cached_property
            def foo(self):
                return {}  # not flyweight

        a = A()
        assert sys.getrefcount(a) == 2  # local here + local in getrefcount

        foo = a.foo
        assert sys.getrefcount(a) == 2  # keep the same
        assert sys.getrefcount(foo) == 3  # + ref in cached_property._instance_to_value

        del a
        assert sys.getrefcount(foo) == 2  # remove from cached_property._instance_to_value

    def test_does_not_cache_if_exception(self):
        side_effect = Mock()

        class A:

            def __init__(self, will_raise):
                self.will_raise = will_raise

            @cached_property
            def foo(self):
                side_effect()
                if self.will_raise:
                    raise ValueError
                return 'foo'

        a = A(will_raise=True)
        with pytest.raises(ValueError):
            a.foo
        assert side_effect.call_count == 1

        with pytest.raises(ValueError):
            a.foo
        assert side_effect.call_count == 2  # has retried

        a.will_raise = False
        assert a.foo == 'foo'
        assert side_effect.call_count == 3

        assert a.foo == 'foo'
        assert side_effect.call_count == 3  # hit

        a.will_raise = True
        # won't retry so won't raise
        assert a.foo == 'foo'
        assert side_effect.call_count == 3  # hit


class A(JSONSerializableMixin):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_config(self):
        return {'x': self.x, 'y': self.y}


class B(JSONSerializableMixin):

    def __init__(self, a):
        self.a = a

    def get_config(self):
        return {'a': self.a.serialize()}

    @classmethod
    def from_config(cls, config_dict):
        return cls(A.deserialize(config_dict['a']))


@pytest.mark.parametrize(
    'cacher, output, filename',
    [
        (PickleCache, {'a': 1, 'b': 2, 'c': [3, 4]}, 'test.pkl'),
        (JSONCache, A(1, 2), 'a.json'),
        (JSONCache, B(A(3, 4)), 'b.json'),
        (NumpyCache, np.random.choice(100, size=[100]), 'test.npz'),
    ],
)
def test_cache_static(tmpdir, cacher, output, filename):
    filename = tmpdir / filename
    create = Mock(return_value=output)

    @cacher.tofile(filename)
    def wrapped_create():
        return create()

    assert equal(wrapped_create(), output)
    assert filename.isfile()  # save to file
    assert create.call_count == 1

    assert equal(wrapped_create(), output)
    assert create.call_count == 1  # load from file, don't create again


def test_cache_format(tmpdir):
    output = '123'
    create = Mock(return_value=output)

    @PickleCache.tofile(tmpdir / "({0}, {1})")
    def wrapped_create(a, b):
        return create()

    assert equal(wrapped_create(1, 2), output)
    assert (tmpdir / "(1, 2)").isfile()  # save to file
    assert create.call_count == 1

    assert equal(wrapped_create(1, 2), output)
    assert create.call_count == 1  # load from file, don't create again
    assert equal(wrapped_create(2, 1), output)
    assert create.call_count == 2  # different key, create again


def test_cache_callable_path(tmpdir):
    output = '123'
    create = Mock(return_value=output)

    @PickleCache.tofile(path=lambda key: tmpdir / key)
    def wrapped_create(key):
        return create()

    assert wrapped_create('a.pkl') == output
    assert (tmpdir / 'a.pkl').isfile()  # save to file
    assert create.call_count == 1

    assert wrapped_create('a.pkl') == output
    assert create.call_count == 1  # load from file, don't create again

    assert wrapped_create('b.pkl') == output
    assert create.call_count == 2  # different key, create again


def test_deserialize_subclass_and_non_subclass():

    class C(JSONSerializableMixin):

        def __init__(self, x):
            self.x = x

        def get_config(self):
            return {'x': self.x}

    class D(C):

        pass

    assert C.deserialize(D(3).serialize()) == D(3)

    with pytest.raises(ValueError):
        D.deserialize(C(1).serialize())


def test_reuse_method_call():
    output = '123'
    mocker = Mock(return_value=output)

    class D:

        def foo(self):
            return mocker()

    d = D()
    with reuse_method_call(d, ['foo']) as new_d:
        assert new_d.foo() == output
        assert mocker.call_count == 1
        assert new_d.foo() == output
        assert mocker.call_count == 1

        assert d.foo() == output
        assert mocker.call_count == 2  # no cache


def equal(x, y):
    return np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y
