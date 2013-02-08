
import functools
import itertools
import math
import operator



# === Exceptions ===

class StatsError(ValueError):
    pass


# === Private utilities ===


def _2nd_moment(data, m):
    """Compensated 2nd moment about m."""
    n = len(data)
    sum_sq = math.fsum((x-m)**2 for x in data)
    sum_comp = math.fsum(x-m for x in data)
    # sum_comp should be zero if the mean is exact, but may not be.
    # For debugging purposes, we secretely expose this.
    # *** THIS IS UNDOCUMENTED AND INTERNAL ONLY ***
    class Float(float): pass
    total = Float(sum_sq - sum_comp**2/n)
    total._comp = sum_comp
    return total


# === Public utilities ===

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


def isnumeric(obj):
    """Return True if obj is a number, otherwise False.

    >> isnumeric(42)
    True
    >> isnumeric([42])
    False

    """
    try:
        obj+0
    except TypeError:
        return False
    else:
        return True


def map_longest(func, *iterables, fillvalue=None):
    """map_longest(func, *iterables [, fillvalue=None]) --> iterator

    Similar to the built-in map function, returns an iterator that computes
    the function using arguments from each of the iterables. ``fillvalue``
    is substituted for any missing values in the iterables.

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

    >>> it = map_strict(lambda a,b: a+b, [1, 10], [2, 30])
    >>> list(it)
    [3, 40]

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
                # Report which iterable was shorter than the others. If
                # there are more than one such short iterable, report the
                # first such.
                p1 = t.index(sentinel)  # The first short iterable.
                # Find the index of the first non-sentinel.
                for i, x in enumerate(t):
                    if x is not sentinel:
                        p2 = i
                        break
                assert p1 != p2
                msg = "iterable %d shorter than iterable %d"
                exception = ValueError(msg % (p1, p2))
            raise exception
        yield func(*t)


# === Tools for vectorizing functions ===

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
            # Bail out early for the scalar case.
            return op(x, y)
    return list(result)


# === Vectorized operators and functions ===

add = functools.partial(apply_op, operator.add)
sub = functools.partial(apply_op, operator.sub)
mul = functools.partial(apply_op, operator.mul)
div = functools.partial(apply_op, operator.truediv)
pow = functools.partial(apply_op, operator.pow)


# === Sums and products ===

def sum(data, start=0):
    """sum(iterable [, start]) -> sum of numbers or columns

    Return a high-precision sum of the given numbers or columns.

    When passed a single sequence or iterator of numbers, ``sum`` adds the
    numbers and returns the total:

    >>> sum([2.25, 4.5, -0.5, 1.0])
    7.25

    If optional argument ``start`` is given, it is added to the total. If
    the iterable is empty, ``start`` (defaulting to 0) is returned.

    When passed an iterable of sequences, each sub-sequence represents a
    row of data, and ``sum`` adds the columns. Each row must have the same
    number of columns, or ValueError is raised. If ``start`` is given, it
    must be either a single number, or a sequence with the same number of
    columns as the data.

    >>> data = [[0, 1, 2, 3],
    ...         [1, 2, 4, 5],
    ...         [2, 3, 6, 7]]
    ...
    >>> sum(data)
    [3, 6, 12, 15]
    >>> sum(data, 1)
    [4, 7, 13, 16]
    >>> sum(data, [1, 0, 1.5, -1.5])
    [4, 6, 13.5, 13.5]

    The numbers are added using high-precision arithmetic that can avoid
    some sources of round-off error:

    >>> sum([1, 1e100, 1, -1e100] * 10000)  # The built-in sum returns zero.
    20000.0

    """
    if isinstance(data, str):
        raise TypeError('data argument cannot be a string')
    if not isinstance(data, list):
        data = list(data)
    if not data:
        return start
    if isiterable(data[0]):
        # Vectorized version of sum.
        total = [42]
    else:
        # Scalar sum.
        total = 42
    return add(total, start)










def mean(data):
    if not isinstance(data, list):
        data = list(data)
    n = len(data)
    total = math.fsum(data)
    if n:
        return total/n
    raise StatsError('mean requires at least one data point')


def pvariance(data, m=None):
    data = list(data)
    if m is None:
        m = mean(data)
    n = len(data)
    return _2nd_moment(data, m)/n



