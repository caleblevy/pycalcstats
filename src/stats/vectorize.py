##  Copyright (c) 2011 Steven D'Aprano.
##  This module is part of the stats package. See the file stats/__init__.py
##  for the licence terms for this software.

"""Vectorized operations and functions.
"""

import functools
import itertools
import math
import operator

_abs = abs



# === Utilities ===


def _is_numeric(obj):
    try:
        obj+0
    except TypeError:
        return False
    else:
        return True


def isiterable(obj):
    """Return True if obj is an iterable, otherwise False.

    >> isiterable([42, 23])
    True
    >> isiterable(42)
    False

    """
    try:
        iter(obj)
    except TypeError:
        return False
    else:
        return True


def map_longest(func, *iterables, fillvalue=None):
    """map_longest(func, *iterables [, fillvalue=None]) --> iterator

    Similar to the built-in map function, returns an iterator that computes
    the function using arguments from each of the iterables. fillvalue is
    substituted for any missing values in the iterables.

    >>> f = lambda a,b,c: (a+b)*c
    >>> it = map_longest(f, [1, 1, 1], [2, 2], [3], fillvalue=100)
    >>> list(it)
    [9, 300, 10100]

    """
    for t in itertools.zip_longest(*iterables, fillvalue=fillvalue):
        yield func(*t)


def map_strict(func, *iterables, exception=None):
    """map_strict(func, *iterables [, exception=None]) --> iterator

    Similar to the built-in map function, returns an iterator that computes
    the function using arguments from each of the iterables.

    >>> it = map_strict(lambda a,b: a+b, [1, 1], [2, 3])
    >>> list(it)
    [3, 4]

    All iterables must be the same length, or exception is raised. If
    exception is None, ValueError is raised.

    >>> it = map_strict(lambda a,b: a+b, [1, 1], [2, 3, 4])
    >>> list(it)  #doctest:+IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
      ...
    ValueError: ...

    """
    sentinel = object()
    for t in itertools.zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in t:
            if exception is None:
                p1 = t.index(sentinel)
                # Find the index of the first non-sentinel.
                for i, x in enumerate(t):
                    if x is not sentinel:
                        p2 = i
                        break
                assert p1 != p2
                msg = "argument %d shorter than argument %d"
                exception = ValueError(msg % (p1, p2))
            raise exception
        yield func(*t)


def assert_(assertion, obj, message=None):
    if __debug__:
        if message is None:
            message = "assertion failed with argument %r"
            getmsg = lambda x: message % x
        else:
            getmsg = lambda x: message
        if isiterable(obj):
            for x in obj:
                assert assertion(x), getmsg(x)
        else:
            assert assertion(obj), getmsg(obj)


# === Tools for vectorizing functions ===


def apply(func, x, *args, assertion=None, **kwargs):
    """Return vectorized func(x, *args, **kwargs).

    If x is a scalar-type (not an iterable), returns:

        func(x, *args, **kwargs)

    >>> f = lambda a,b,c: c*(a+b)
    >>> apply(f, 3, 1, 2)
    8

    If x is a vector-type (an iterable), the function call is treated as an
    operation on columns. A list of results is returned:

        [func(x0, ...), func(x1, ...), func(x2, ...), ...]

    >>> apply(f, [3, 5], [1, 3], [1, 2])
    [4, 16]

    Except as described below, each argument must have the same length, or
    ValueError is raised. The exception is that scalar positional arguments
    are automatically expanded to meet the required length:

    >>> apply(f, [2, 3, 2, 3], 2, [1, 2, 3, 4])
    [4, 10, 12, 20]

    If optional argument ``assertion`` is not None, it should be a function
    which takes a single argument. If x is a scalar, the result is passed
    to the assertion function and then asserted; if x is a vector, instead
    the assertion function is called with each element of result.

    >>> small_enough = lambda obj: len(obj) < 5
    >>> apply(lambda a,b: a*b, ['a', 'b'], 3, assertion=small_enough)
    ['aaa', 'bbb']
    >>> #doctest:+IGNORE_EXCEPTION_DETAIL
    ... apply(lambda a,b: a*b, ['a', 'bb'], 3, assertion=small_enough)
    Traceback (most recent call last):
      ...
    AssertionError: ...

    Note: if func also takes a keyword argument ``assertion``, it is shadowed
    by the keyword argument of the same name and cannot be supplied.
    """
    f = functools.partial(func, **kwargs)
    if isiterable(x):
        # Vectorized function call.
        x = tuple(x)
        n = len(x)
        args = [list(a) if isiterable(a) else [a]*n for a in args]
        result = list(map_strict(f, x, *args))
    else:
        # Scalar function call.
        result = func(x, *args, **kwargs)
    if assertion is not None:
        assert_(assertion, result)
    return result


def apply_op(op, x, y):
    """Return vectorized ``x op y``."""
    if isiterable(x):
        if isiterable(y):
            result = map_strict(op, x, y)
        else:
            result = map(op, x, itertools.repeat(y))
    else:
        if isiterable(y):
            result = map(op, itertools.repeat(x), y)
        else:
            # Technically, this is a scalar operation. Bail out early.
            return op(x, y)
    return list(result)


# === Vectorized operators and functions ===


add = functools.partial(apply_op, operator.add)
div = functools.partial(apply_op, operator.truediv)
mul = functools.partial(apply_op, operator.mul)
pow = functools.partial(apply_op, operator.pow)
sub = functools.partial(apply_op, operator.sub)

abs = lambda x: apply(_abs, x)
sqr = lambda x: apply(lambda x: x*x, x)
sqrt = lambda x: apply(math.sqrt, x)


# Low precision sum, but should work on anything that supports + operator.
def sum(x, start=0):
    return functools.reduce(add, x, start)


def prod(x, start=1):
    return functools.reduce(mul, x, start)


def fsum(x, start=0):
    def add(a, b):
        try:
            return a+b
        except TypeError:
            if _is_numeric(a) and _is_numeric(b):
                # Downgrade both to floats and try again.
                return float(a) + float(b)
            raise
    add = functools.partial(apply_op, add)
    return functools.reduce(add, x, start)
