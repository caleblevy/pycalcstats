#!/usr/bin/env python3

##  Package stats.py
##
##  Copyright (c) 2011 Steven D'Aprano.
##
##  Permission is hereby granted, free of charge, to any person obtaining
##  a copy of this software and associated documentation files (the
##  "Software"), to deal in the Software without restriction, including
##  without limitation the rights to use, copy, modify, merge, publish,
##  distribute, sublicense, and/or sell copies of the Software, and to
##  permit persons to whom the Software is furnished to do so, subject to
##  the following conditions:
##
##  The above copyright notice and this permission notice shall be
##  included in all copies or substantial portions of the Software.
##
##  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
##  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
##  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
##  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
##  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
##  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
##  SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


"""
Statistics package for Python 3.

The statistics functions are divided up into separate modules:

    Module              Description
    ==================  =============================================
    stats               Basic calculator statistics.
    stats.co            Coroutine versions of selected functions.
    stats.multivar      Multivariate (multiple variable) statistics.
    stats.order         Order statistics.
    stats.univar        Univariate (single variable) statistics.
    stats.vectorize     Utilities for vectorizing functions.

For further details, see the individual modules.


The ``stats`` module provides nine statistics functions:

    Function        Description
    ==============  =============================================
    mean*           Arithmetic mean (average) of data.
    minmax          Minimum and maximum of the arguments.
    product*        Product of data.
    pstdev*         Population standard deviation of data.
    pvariance*      Population variance of data.
    running_product Running product coroutine.
    running_sum     High-precision running sum coroutine.
    stdev*          Sample standard deviation of data.
    sum*            High-precision sum of data.
    variance*       Sample variance of data (bias-corrected).

Functions marked with * can operate on columnar data (see below).

The module also includes two public utility functions plus an exception
class used for some statistical errors:

    Name            Description
    ==============  =============================================
    add_partial     Utility for performing high-precision sums.
    coroutine       Utility for initialising coroutines.
    StatsError      Subclass of ValueError.


Examples
--------

>>> import stats
>>> stats.mean([-1.0, 2.5, 3.25, 5.75])
2.625
>>> stats.stdev([2.5, 3.25, 5.5, 11.25, 11.75])  #doctest: +ELLIPSIS
4.38961843444...
>>> stats.minmax(iter([19, 23, 15, 42, 31]))
(15, 42)


Columnar data
-------------

As well as operating on a single row, as in the examples above, the functions
marked with * can operate on data in columns:

>>> data = [[0, 1, 1, 2],  # row 1, 4 columns
...         [1, 1, 2, 4],  # row 2
...         [2, 1, 3, 8]]  # row 3
...
>>> stats.sum(data)  # Sum each column.
[3, 3, 6, 14]
>>> stats.variance(data)  #doctest: +ELLIPSIS
[1.0, 0.0, 1.0, 9.333333333333...]

For further details, see the individual functions.

"""

# Package metadata.
__version__ = "0.2.0a"
__date__ = "2011-03-?????????????????????????????????????????????????"
__author__ = "Steven D'Aprano"
__author_email__ = "steve+python@pearwood.info"


__all__ = [ 'add_partial', 'coroutine', 'mean', 'minmax', 'product',
            'pstdev', 'pvariance', 'running_sum', 'StatsError', 'stdev',
            'sum', 'variance',
          ]


import collections
import functools
import itertools
import math
import operator

from builtins import sum as _sum

import stats.vectorize as v



# === Exceptions ===

class StatsError(ValueError):
    pass


# === Public utilities ===

def coroutine(func):
    """Decorator to prime coroutines when they are initialised."""
    @functools.wraps(func)
    def started(*args, **kwargs):
        cr = func(*args,**kwargs)
        cr.send(None)
        return cr
    return started


# Modified from http://code.activestate.com/recipes/393090/
# Thanks to Raymond Hettinger.
def add_partial(x, partials):
    """Helper function for full-precision summation of binary floats.

    Adds finite (not NAN or INF) x in place to the list partials.

    Example usage:

    >>> partials = []
    >>> add_partial(1e100, partials)
    >>> add_partial(1e-100, partials)
    >>> add_partial(-1e100, partials)
    >>> partials
    [1e-100, 0.0]

    Initialise partials to be a list containing at most one finite float
    (i.e. no INFs or NANs). Then for each float you wish to add, call
    ``add_partial(x, partials)``.

    When you are done, call sum(partials) to round the summation to the
    precision supported by float.

    If you initialise partials with more than one value, or with special
    values (NANs or INFs), results are undefined.
    """
    # Rounded x+y stored in hi with the round-off stored in lo.  Together
    # hi+lo are exactly equal to x+y.  The inner loop applies hi/lo summation
    # to each partial so that the list of partial sums remains exact.
    # Depends on IEEE-754 arithmetic guarantees.  See proof of correctness at:
    # www-2.cs.cmu.edu/afs/cs/project/quake/public/papers/robust-arithmetic.ps
    i = 0
    for y in partials:
        if abs(x) < abs(y):
            x, y = y, x
        hi = x + y
        lo = y - (hi - x)
        if lo:
            partials[i] = lo
            i += 1
        x = hi
    partials[i:] = [x]


# === Private utilities ===

class _countiter:
    """Iterator that counts how many elements it has seen.

    >>> c = _countiter(['a', 1, None, 'c'])
    >>> _ = list(c)
    >>> c.count
    4

    """
    def __init__(self, iterable):
        self.it = iter(iterable)
        self.count = 0
    def __next__(self):
        x = next(self.it)
        self.count += 1
        return x
    def __iter__(self):
        return self


def _is_numeric(obj):
    """Return True if obj is a number, otherwise False.

    >>> _is_numeric(2.5)
    True
    >>> _is_numeric('spam')
    False

    """
    try:
        obj + 0
    except TypeError:
        return False
    else:
        return True


class _Adder:
    """High precision addition."""

    def __init__(self, partials=None):
        if partials is None:
            partials = []
        self.partials = partials

    def add(self, x):
        """Add numeric value x to self.partial."""
        # Handle special values:
        #
        #   |   x   |   y   |  y+x  |   where y = partials
        #   +-------+-------+-------|         * = any finite value
        #   |   ?   |   []  |   x   |         ? = any value, finite or not
        #   |  NAN  |   ?   |  NAN  |         [] = empty partials list
        #   |   ?   |  NAN  |  NAN  |
        #   |  INF  |  INF  |  INF  | <= same sign
        #   | +INF  | -INF  |  NAN  | <= opposite signs
        #   | -INF  | +INF  |  NAN  |
        #   |   *   |  INF  |  INF  |
        #   |  INF  |   *   |  INF  |
        #
        partials = self.partials[:]  # Make a copy.
        if not partials:
            # nothing + anything
            partials = [x]
        elif math.isnan(x):
            # anything + NAN = NAN
            partials = [x]  # Latest NAN beats previous NAN (if any).
        else:
            y = partials[0]
            if math.isnan(y):
                # NAN + anything = NAN.
                pass
            elif math.isinf(y):
                if math.isinf(x):
                    if float(x) == float(y):
                        # INFs have the same sign.
                        assert (x > 0) == (y > 0)
                        partials = [x]  # Latest INF wins.
                    else:
                        # INFs have opposite sign.
                        assert (x > 0) != (y > 0)
                        partials = [type(x)('nan')]
                else:
                    # INF + finite = INF
                    assert not math.isnan(x)  # Handled earlier.
            elif math.isinf(x):
                # finite + INF = INF
                assert not math.isnan(y)  # Handled earlier.
                assert not math.isinf(y)
                partials = [x]
            else:
                # finite + finite
                try:
                    add_partial(x, partials)
                except TypeError:
                    # This probably means we're trying to add Decimal to
                    # float. Coerce to floats and try again.
                    partials[:] = map(float, partials)
                    x = float(x)
                    add_partial(x, partials)
        return type(self)(partials)

    def value(self):
        return _sum(self.partials)


_add = functools.partial(v.apply_op, _Adder.add)

def _fsum(x, start=0):
    return functools.reduce(_add, x, _Adder([start]))


def _len_sum(iterable, func=None):
    """\
    _len_sum(iterable) -> len(iterable), sum(iterable)
    _len_sum(iterable, func) -> len(iterable), sum(func(items of data))

    Return a two-tuple of the length of data and the sum of func() of the
    items of data. If func is None (the default)), use just the sum of items
    of data.
    """
    # Special case for speed.
    if isinstance(iterable, list):
        n = len(iterable)
    else:
        n = None
        iterable = _countiter(iterable)
    if func is None:
        total = _fsum(iterable)
    else:
        total = _fsum(func(x) for x in iterable)
    if n is None:
        n = iterable.count
    if isinstance(total, list):
        total = [t.value() for t in total]
    else:
        total = total.value()
    return (n, total)


def _std_moment(data, m, s, r):
    """Return the length and standardised moment of order r = 1...4."""
    assert r in (1, 2, 3, 4), "private function not intended for r != 1...4"
    if m is None or s is None:
        # We need multiple passes over the data, so make sure we can.
        if not isinstance(data, list):
            data = list(data)
        if m is None: m = mean(data)
        if s is None: s = pstdev(data, m)
    # Minimize the number of arithmetic operations needed for some
    # common functions.
    if False and s == 1:  # FIXME this optimization is currently disabled.
        if r == 1:
            args = (m,)
            f = lambda x, m: (x-m)
        elif r == 2:
            args = (m,)
            f = lambda x, m: (x-m)**2
        else:
            args = (m, r)
            f = lambda x, m, r: (x-m)**r
    else:
        args = (m, s, r)
        f = lambda x, m, s, r: ((x-m)/s)**r
    n, total = _len_sum(v.apply(f, x, *args) for x in data)
    return (n, total)
    # FIXME the above may not be accurate enough for 2nd moments (x-m)**2
    # A more accurate algorithm is the compensated version:
    #   sum2 = sum((x-m)**2) as above
    #   sumc = sum(x-m)  # Should be zero, but may not be.
    #   total = sum2 - sumc**2/n


# === Sums and products ===

def sum(data, start=0):
    """sum(iterable_of_numbers [, start]) -> sum of numbers
    sum(iterable_of_rows [, start]) -> sums of columns

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
    # Calculate the length and sum of data.
    count, total = _len_sum(data)
    if not count:
        return start
    # Add start as needed.
    return v.add(total, start)


@coroutine
def running_sum(start=None):
    """Running sum co-routine.

    With no arguments, ``running_sum`` consumes values and returns the
    running sum of arguments sent to it:

    >>> rsum = running_sum()
    >>> rsum.send(1)
    1
    >>> [rsum.send(n) for n in (2, 3, 4)]
    [3, 6, 10]

    If optional argument ``start`` is given and is not None, it is used as
    the initial value for the running sum:

    >>> rsum = running_sum(9)
    >>> [rsum.send(n) for n in (1, 2, 3)]
    [10, 12, 15]

    """
    if start is None: start = []
    else: start = [start]
    total = _Adder(start)
    x = (yield None)
    while True:
        total = total.add(x)
        x = (yield total.value())


@coroutine
def running_product(start=None):
    """Running product co-routine.

    With no arguments, ``running_product`` consumes values and returns the
    running product of arguments sent to it:

    >>> rp = running_product()
    >>> rp.send(1)
    1
    >>> [rp.send(n) for n in (2, 3, 4)]
    [2, 6, 24]

    If optional argument ``start`` is given and is not None, it is used as
    the initial value for the running product:

    >>> rp = running_product(9)
    >>> [rp.send(n) for n in (1, 2, 3)]
    [9, 18, 54]

    """
    if start is not None:
        total = start
    else:
        total = 1
    x = (yield None)
    while True:
        try:
            total *= x
        except TypeError:
            if not _is_numeric(x):
                raise
            # Downgrade to floats and try again.
            x = float(x)
            total = float(total)
            continue
        x = (yield total)


def product(data, start=1):
    """product(iterable_of_numbers [, start]) -> product of numbers
    product(iterable_of_rows [, start]) -> product of columns

    Return the product of the given numbers or columns.

    When passed a single sequence or iterator of numbers, ``product``
    multiplies the numbers and returns the total:

    >>> product([2.25, 4.5, -0.5, 10])
    -50.625
    >>> product([1, 2, -3, 2, -1])
    12

    If optional argument ``start`` is given, it is multiplied to the total.
    If the iterable is empty, ``start`` (defaulting to 1) is returned.

    When passed an iterable of sequences, each sub-sequence represents a
    row of data, and product() multiplies each column. Each row must have
    the same number of columns, or ValueError is raised. If ``start`` is
    given, it must be either a single number, or a sequence with the same
    number of columns as the data.

    >>> data = [[0, 1, 2, 3],
    ...         [1, 2, 4, 6],
    ...         [2, 3, 6, 0.5]]
    ...
    >>> product(data)
    [0, 6, 48, 9.0]
    >>> product(data, 2)
    [0, 12, 96, 18.0]
    >>> product(data, [2, 1, 0.25, -1.5])
    [0, 6, 12.0, -13.5]

    """
    if isinstance(data, str):
        raise TypeError('data argument cannot be a string')
    return v.prod(data, start)
    # Note: do *not* be tempted to do something clever with logarithms:
    #   math.exp(sum([math.log(x) for x in data], start))
    # is FAR less accurate than naive multiplication.


# === Basic univariate statistics ===

def mean(data):
    """mean(iterable_of_numbers) -> arithmetic mean of numbers
    mean(iterable_of_rows) -> arithmetic means of columns

    Return the arithmetic mean of the given numbers or columns.

    The arithmetic mean is the sum of the data divided by the number of data
    points. It is commonly called "the average", although it is actually only
    one of many different mathematical averages. It is a measure of the
    central location of the data.

    When passed a single sequence or iterator of numbers, ``mean`` adds the
    data points and returns the total divided by the number of data points:

    >>> mean([1.0, 2.0, 3.0, 4.0])
    2.5

    When passed an iterable of sequences, each inner sequence represents a
    row of data, and ``mean`` returns the mean of each column. The rows must
    have the same number of columns, or ValueError is raised.

    >>> data = [[0, 1, 2, 3],
    ...         [1, 2, 4, 5],
    ...         [2, 3, 6, 7]]
    ...
    >>> mean(data)
    [1.0, 2.0, 4.0, 5.0]

    The sample mean is an unbiased estimator of the true population mean.
    However, the mean is strongly effected by outliers and is not a robust
    estimator for central location: the mean is not necessarily a typical
    example of the data points.
    """
    count, total = _len_sum(data)
    if not count:
        raise StatsError('mean of empty sequence is not defined')
    return v.div(total, count)


def variance(data, m=None):
    """variance(iterable_of_numbers [, m]) -> sample variance of numbers
    variance(iterable_of_rows [, m]) -> sample variance of columns

    Return the unbiased sample variance of the given numbers or columns.
    The variance is a measure of the variability (spread or dispersion) of
    data. A large variance indicates that the data is spread out; a small
    variance indicates it is clustered closely around the central location.

        WARNING: The mathematical terminology related to variance is
        often inconsistent and confusing. This is the variance with
        Bessel's correction for bias, also known as variance with N-1
        degrees of freedom. See Wolfram Mathworld for further details:

        http://mathworld.wolfram.com/Variance.html
        http://mathworld.wolfram.com/SampleVariance.html

    When given a single iterable of data, ``variance`` returns the sample
    variance of that data:

    >>> variance([3.5, 2.75, 1.75, 1.25, 1.25,
    ...           0.5, 0.25])  #doctest: +ELLIPSIS
    1.37202380952...

    If you already know the mean of your data, you can supply it as the
    optional second argument ``m``:

    >>> data = [0.0, 0.25, 0.25, 1.25, 1.5, 1.75, 2.75, 3.25]
    >>> m = mean(data)  # Save the mean for later use.
    >>> variance(data, m)  #doctest: +ELLIPSIS
    1.42857142857...

        CAUTION: "Garbage in, garbage out" applies. If the value you
        supply as ``m`` is not the mean for your data, the result
        returned may not be statistically valid.

    If the argument to ``variance`` is an iterable of sequences, each inner
    sequence represents a row of data, and ``variance`` returns the variance
    of each column. Each row must have exactly the same number of columns, or
    ValueError will be raised.

    >>> data = [[0, 1, 2],
    ...         [1, 1, 3],
    ...         [1, 2, 5],
    ...         [2, 4, 6]]
    ...
    >>> variance(data)  #doctest: +ELLIPSIS
    [0.6666666666..., 2.0, 3.3333333333...]

    If ``m`` is given for such columnar data, it must be either a single
    number, or a sequence with the same number of columns as the data.

    See also ``pvariance``.
    """
    return _variance(data, m, 1)


def stdev(data, m=None):
    """stdev(iterable_of_numbers [, m]) -> standard deviation of numbers
    stdev(iterable_of_rows [, m]) -> standard deviation of columns

    Returns the sample standard deviation (with N-1 degrees of freedom)
    of the given numbers or columns. The standard deviation is the square
    root of the variance.

    Optional argument ``m`` has the same meaning as for ``variance``.

    >>> stdev([1.5, 2.5, 2.5, 2.75, 3.25, 4.75])  #doctest: +ELLIPSIS
    1.08108741552...

    >>> data = [[0, 1, 2],
    ...         [1, 1, 3],
    ...         [1, 2, 5],
    ...         [2, 4, 6]]
    ...
    >>> stdev(data)  #doctest: +ELLIPSIS
    [0.816496580927..., 1.41421356237..., 1.82574185835...]

    Note that although ``variance`` is an unbiased estimate for the
    population variance, ``stdev`` itself is *not* unbiased.
    """
    svar = variance(data, m)
    return v.sqrt(svar)


def pvariance(data, m=None):
    """pvariance(iterable_of_numbers [, m]) -> population variance of numbers
    pvariance(iterable_of_rows [, m]) -> population variance of columns

    Return the population variance of the given numbers or columns. The
    variance is a measure of the variability (spread or dispersion) of
    data. A large variance indicates that the data is spread out; a small
    variance indicates it is clustered closely around the central location.

    See ``variance`` for further information.

    If your data represents the entire population, you should use this
    function. If your data is a sample of the population, this function
    returns a biased estimate of the variance. For an unbiased estimate,
    use ``variance`` instead.

    Calculate the variance of populations:

    >>> pvariance([0.0, 0.25, 0.25, 1.25, 1.5, 1.75, 2.75, 3.25])
    1.25

    If the argument to ``pvariance`` is an iterable of sequences, each inner
    sequence represents a row of data, and the variance of each column is
    returned:

    >>> data = [[0, 1, 2, 3, 4],
    ...         [1, 1, 3, 5, 4],
    ...         [1, 2, 5, 5, 7],
    ...         [2, 4, 6, 7, 9]]
    ...
    >>> pvariance(data)
    [0.5, 1.5, 2.5, 2.0, 4.5]

    Each row must have exactly the same number of columns, or ValueError
    will be raised.

    If you already know the mean, you can pass it to ``pvariance`` as the
    optional second argument ``m``. For columnar data, ``m`` must be either
    a single number, or it must contain the same number of columns as the
    data.
    """
    return _variance(data, m, 0)


def pstdev(data, m=None):
    """pstdev(iterable_of_numbers [, m]) -> population std dev of numbers
    pstdev(iterable_of_rows [, m]) -> population std dev of columns

    Returns the population standard deviation (with N degrees of freedom)
    of the given numbers or columns. The standard deviation is the square
    root of the variance.

    Optional argument ``m`` has the same meaning as for ``pvariance``.

    >>> pstdev([1.5, 2.5, 2.5, 2.75, 3.25, 4.75])  #doctest: +ELLIPSIS
    0.986893273527...

    >>> data = [[0, 1, 2],
    ...         [1, 1, 3],
    ...         [1, 2, 5],
    ...         [2, 4, 6]]
    ...
    >>> pstdev(data)  #doctest: +ELLIPSIS
    [0.707106781186..., 1.22474487139..., 1.58113883008...]

    """
    pvar = pvariance(data, m)
    return v.sqrt(pvar)


def _variance(data, m, p):
    """Return an estimate of variance with N-p degrees of freedom."""
    n, ss = _std_moment(data, m, 1, 2)
    assert n >= 0
    if n <= p:
        raise StatsError(
        'at least %d items are required but only got %d' % (p+1, n))
    den = n - p
    v.assert_(lambda x: x >= 0.0, ss)
    return v.div(ss, den)


def minmax(*values, **kw):
    """minmax(iterable [, key=func]) -> (minimum, maximum)
    minmax(a, b, c, ... [, key=func]) -> (minimum, maximum)

    With a single iterable argument, return a two-tuple of its smallest and
    largest items. With two or more arguments, return the smallest and
    largest arguments. ``minmax`` is similar to the built-ins ``min`` and
    ``max``, but can return the two items with a single pass over the data,
    allowing it to work with iterators.

    >>> minmax([3, 2, 1, 6, 5, 4])
    (1, 6)
    >>> minmax(4, 5, 6, 1, 2, 3)
    (1, 6)

    The optional keyword-only argument ``key`` specifies a key function:

    >>> minmax('aaa', 'bbbb', 'c', 'dd', key=len)
    ('c', 'bbbb')

    """
    if len(values) == 0:
        raise TypeError('minmax expected at least one argument, but got none')
    elif len(values) == 1:
        values = values[0]
    if list(kw.keys()) not in ([], ['key']):
        raise TypeError('minmax received an unexpected keyword argument')
    if isinstance(values, collections.Sequence):
        # For speed, fall back on built-in min and max functions when
        # data is a sequence and can be safely iterated over twice.
        # TODO this could be unnecessary if this were re-written in C.
        minimum = min(values, **kw)
        maximum = max(values, **kw)
        # The number of comparisons is N-1 for both min() and max(), so the
        # total used here is 2N-2, but performed in fast C.
    else:
        # Iterator argument, so fall back on a slow pure-Python solution
        # that calculates the min and max lazily. Even if values is huge,
        # this should work.
        # Note that the number of comparisons is 3*ceil(N/2), which is
        # approximately 50% fewer than used by separate calls to min & max.
        key = kw.get('key')
        if key is not None:
            it = ((key(value), value) for value in values)
        else:
            it = ((value, value) for value in values)
        try:
            keyed_min, minimum = next(it)
        except StopIteration:
            # Don't directly raise an exception inside the except block,
            # as that exposes the StopIteration to the caller. That's an
            # implementation detail that should be avoided. See PEP 3134
            # http://www.python.org/dev/peps/pep-3134/
            # and specifically the open issue "Suppressing context".
            empty = True
        else:
            empty = False
        if empty:
            raise ValueError('minmax argument is empty')
        keyed_max, maximum = keyed_min, minimum
        try:
            while True:
                a = next(it)
                try:
                    b = next(it)
                except StopIteration:
                    b = a
                if a[0] > b[0]:
                    a, b = b, a
                if a[0] < keyed_min:
                    keyed_min, minimum = a
                if b[0] > keyed_max:
                    keyed_max, maximum = b
        except StopIteration:
            pass
    return (minimum, maximum)

