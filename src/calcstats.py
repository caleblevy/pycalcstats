#!/usr/bin/env python3

##  Module calcstats.py
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


"""\
=====================
Calculator statistics
=====================

Simple calculator-style statistics.

This module provides the following statistics and related functions:

    Statistics          Description
    ==================  ===============================================
    mean                Arithmetic mean (average) of data.
    minmax              Minimum and maximum of the arguments.
    product             Product of data.
    pstdev              Population standard deviation of data.
    pvariance           Population variance of data.
    running_mean        Running average of data.
    running_product     Running product of data.
    running_sum         High-precision running sum of data.
    stdev               Sample standard deviation of data.
    sum                 High-precision sum of data.
    variance            Sample variance of data (bias-corrected).
    welford             Running sum of square residuals.


    Utilities           Description
    ==================  ===============================================
    add_partial         Helper function for high-precision addition.
    coroutine           Decorator to initialise co-routines.
    StatsError          Exception raised on invalid statistics.


Examples
--------

>>> data = [2, 0, 3, 2, 5, 6, 1, 2, 3, 2, 1, 2]
>>> mean(data)  #doctest: +ELLIPSIS
2.41666666666...
>>> stdev(data)  #doctest: +ELLIPSIS
1.67648622440...

"""

# No support for missing values, or arrays (at this time).
# Behaviour with NANs and INFs is officially undefined (at this time).
# Full support for ints, fractions, decimals, and floats.
# Behaviour with mixed data types (e.g. float + Decimal) is undefined.


import collections
import functools
import itertools
import math
import numbers
import operator

from builtins import sum as _builtin_sum


# Package metadata.
__version__ = "0.1a"
__date__ = "2011-09-09"
__author__ = "Steven D'Aprano"
__author_email__ = "steve+python@pearwood.info"

__all__ = [ 'add_partial', 'coroutine', 'mean', 'minmax', 'product',
            'pstdev', 'pvariance', 'running_mean', 'running_product',
            'running_sum', 'StatsError', 'stdev', 'sum', 'variance',
            'welford',
          ]


# === Exceptions ===

class StatsError(ValueError):
    pass


# === Utilities ===

def coroutine(func):
    """Decorator to prime coroutines when they are initialised."""
    @functools.wraps(func)
    def started(*args, **kwargs):
        cr = func(*args,**kwargs)
        cr.send(None)
        return cr
    return started


def add_partial(partials, x):
    """Helper function for full-precision summation.

    Arguments:

        partials    list containing numeric partial sums
        x           number to add in-place to partials

    ``add_partials`` supports high-preciation summation of floats and
    Decimals. It can also be used for ints and Fractions, although there
    is little benefit in doing so since addition of those is already exact.

    Non-finite float and Decimal values (NANs and INFs) and complex
    numbers are not supported. The behaviour of addition with mixed
    float/Fraction/Decimal is implementation-dependent.

    Usage:
        * Initialise partials to be a list of at most a single number.
        * Call ``add_partial(partials, x)`` for each value you want to add.
        * Call built-in ``sum(partials)`` to reduce the list of partial
          sums to a single number (or otherwise add the partial sums).

    Example:

    >>> partials = []
    >>> add_partial(partials, 1e100)
    >>> add_partial(partials, 1e-100)
    >>> add_partial(partials, -1e100)
    >>> partials
    [1e-100, 0.0]

    Rounding is done once, when you sum the partials list, rather than after
    every addition.
    """
    # Based on Raymond Hettinger's recipe
    # http://code.activestate.com/recipes/393090/
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


# === Sums ===

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
    if start is None:
        partials = []
    else:
        partials = [start]
    x = (yield None)
    while True:
        add_partial(partials, x)
        x = (yield _builtin_sum(partials))


def sum(data, start=0):
    """sum(iterable [, start]) -> sum of numbers

    Return a high-precision sum of ``data``, a sequence or iterator
    of numbers:

    >>> sum([2.25, 4.5, -0.5, 1.0])
    7.25

    If optional argument ``start`` is given and is not None, it is added
    to the total. If ``data`` is empty, ``start`` (defaulting to 0) is
    returned.

    The summation is done using high-precision arithmetic that can avoid
    some sources of round-off error:

    >>> sum([1, 1e100, 1, -1e100] * 10000)  # The built-in sum returns zero.
    20000.0

    """
    rs = running_sum(start)
    total = start
    for x in data:
        total = rs.send(x)
    return total


# === Products ===

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
    if start is None:
        total = 1
    else:
        total = start
    x = (yield None)
    while True:
        total *= x
        x = (yield total)


def product(data, start=1):
    """product(iterable [, start]) -> product of numbers

    Return the product of ``data``, a sequence or iterator of numbers:

    >>> product([2.0, 1.5, 3.0, 0.25])
    2.25

    If optional argument ``start`` is given and is not None, the total is
    multiplied by it. If ``data`` is empty, ``start`` (defaulting to 1) is
    returned.
    """
    return functools.reduce(operator.mul, data, start)
    # Don't be tempted to do anything clever with summation of logarithms,
    # since that ends up *much* less accurate.


# === Means ===

@coroutine
def running_mean():
    """Running mean co-routine.

    ``running_mean()`` consumes values and returns the running average:

    >>> aver = running_mean()
    >>> aver.send(1)
    1.0
    >>> [aver.send(n) for n in (2, 3, 4)]
    [1.5, 2.0, 2.5]

    The running average, also known as the cumulative moving average,
    takes data one item at a time:

        a, b, c, d, ...

    and returns the values:

        a, (a+b)/2, (a+b+c)/3, (a+b+c+d)/4, ...

    >>> aver = running_mean()
    >>> [aver.send(n) for n in (40, 30, 50, 46, 39, 44)]
    [40.0, 35.0, 40.0, 41.5, 41.0, 41.5]

    """
    n = 0
    rs = running_sum()
    x = (yield None)
    while True:
        total = rs.send(x)
        n += 1
        x = (yield total/n)


def mean(data):
    """mean(iterable) -> arithmetic mean of numbers

    Returns the arithmetic mean ("the average") of ``data``.

    >>> mean([1.0, 2.0, 3.0, 4.0])
    2.5

    The mean of a data sample is a measure of the central location of the
    data. It is an unbiased estimator for the true population mean. However,
    the mean is strongly effected by outliers and is not a robust estimator
    for central location: the mean is not necessarily a typical example of
    the data points:

    >>> mean([1]*99 + [10001])
    101.0

    """
    n = 0
    rs = running_sum()
    for n, x in enumerate(data, 1):
        total = rs.send(x)
    if not n:
        raise StatsError('mean of empty sequence is not defined')
    return total/n


# === Variance and standard deviation ===

# The variance of a population, theta-squared, is defined as:
#
#   σ2 = 1/n * Σ(x - µ)**2
#
# where the summation is over the entire set of all x, µ is the mean of
# the entire population, and n is the size of the population. This is
# mathematically equivalent to the so-called "computational formula for
# the variance":
#
#   σ2 = 1/n**2 * (n*Σ(x**2) - (Σx)**2)
#
# Although this second form is well-suited to exact computation by hand for
# small data sets, it should otherwise be avoided in practice as it is prone
# to catastrophic cancellation, leading to inaccurate answers and sometimes
# even impossible (negative) results.


@coroutine
def welford():
    """Coroutine implementation of Welford's method of calculating variance.

    Returns the running sum of squared residuals sum((x-m)**2), where
    m is the mean of the values seen so far.

    >>> rs = welford()
    >>> rs.send(2.0)
    0.0
    >>> [rs.send(x) for x in (3.0, 4.0, 2.0)]
    [0.5, 2.0, 2.75]

    To convert the values returned by this into running variances, divide by
    n (the item number) or n-1.

        Note: for more accurate results, use this on the residues
        (x - mu) instead of the raw x values, with mu = the population
        mean (if known) or the sample mean.

    """
    rs = running_sum()
    x = (yield None)
    m = x  # First estimate of the mean is the first value.
    i = 1
    while True:
        delta = x - m
        m += delta/i  # Update the mean.
        total = rs.send(delta*(x-m))  # Update the sum of squared residuals.
        assert total >= 0
        x = (yield total)
        i += 1


def _comp_var(data, p):
    """So-called "computational formula for variance".

    Implements the variance formula
    σ2 = 1/n**2 * (n*Σ(x**2) - (Σx)**2)

    FOR TESTING AND COMPARISON USE ONLY, DO NOT USE IN PRODUCTION.
    This formula is numerically unstable and can be extremely inaccurate,
    possibly as bad as returning negative results.
    """
    n = len(data)
    s1 = _builtin_sum(x**2 for x in data)
    s2 = _builtin_sum(data)
    return (n*s1 - s2**2)/(n*(n-p))


def _variance(data, mu, p):
    """Return an estimate of variance relative to population mean mu
    using N-p degrees of freedom.

    Requires data to be a sequence.
    """
    if mu is None:
        # First pass over data to calculate the mean.
        mu = mean(data)
    n = len(data)
    if n <= p:
        raise StatsError(
            'at least %d items are required but only got %d' % (p+1, n))
    sum_squares = sum((x-mu)**2 for x in data)
    sum_residues = sum(x-mu for x in data)
        # sum_residues should be zero, if the values are infinitely precise.
        # But because neither float nor Decimal are, sum_residues may not be
        # zero, and we use it to calculate the compensated sum of squares
        # which should be more accurate than sum_squares alone.
    total = sum_squares - sum_residues**2/n
    assert total >= 0
    return total/(n-p)


def variance(data, m=None):
    """variance(iterable_of_numbers [, m]) -> sample variance of numbers

    Returns the unbiased sample variance of ``data``.

    The variance is a measure of the variability (spread or dispersion) of
    data. A large variance indicates that the data is spread out; a small
    variance indicates it is clustered closely around the central location.

        A note on terminology
        ---------------------
        The mathematical terminology related to variance is often
        inconsistent and confusing. This is the variance with Bessel's
        correction for bias, also known as variance with N-1 degrees
        of freedom. See Wolfram Mathworld for further details:

        http://mathworld.wolfram.com/Variance.html
        http://mathworld.wolfram.com/SampleVariance.html

    Arguments:

        data    sequence of numeric values
        m       (optional) None, or the mean of data

    When given a single sequence  of data, ``variance`` returns the sample
    variance of that data:

    >>> variance([3.5, 2.75, 1.75, 1.25, 1.25,
    ...           0.5, 0.25])  #doctest: +ELLIPSIS
    1.37202380952...

    If you already know the sample mean of your data, you can supply it as
    the optional second argument ``m``:

    >>> data = [0.0, 0.25, 0.25, 1.25, 1.5, 1.75, 2.75, 3.25]
    >>> m = mean(data)  # Save the mean for later use.
    >>> variance(data, m)  #doctest: +ELLIPSIS
    1.42857142857...

        CAUTION: "Garbage in, garbage out" applies. If the value you
        supply as ``m`` is not actually the mean for your data, the
        result returned may not be statistically valid.

    See also ``pvariance``.
    """
    return _variance(data, m, 1)


def stdev(data, m=None):
    """stdev(sequence_of_numbers [, m]) -> standard deviation of numbers

    Returns the sample standard deviation (with N-1 degrees of freedom)
    of the given numbers. The standard deviation is the square root of
    the variance.

    Optional argument ``m`` has the same meaning as for ``variance``.

    >>> stdev([1.5, 2.5, 2.5, 2.75, 3.25, 4.75])  #doctest: +ELLIPSIS
    1.08108741552...

    Note that although ``variance`` is an unbiased estimate for the
    population variance, ``stdev`` itself is not unbiased.
    """
    svar = variance(data, m)
    return math.sqrt(svar)


def pvariance(data, m=None):
    return _variance(data, m, 0)


def pstdev(data, m=None):
    sigma = pvariance(data, m)
    assert sigma >= 0
    return math.sqrt(sigma)



# === Other statistics functions ===

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
        minimum = min(values, **kw)
        maximum = max(values, **kw)
        # The number of comparisons is N-1 for both min() and max(), so the
        # total used here is 2N-2, but performed in fast C.
    else:
        # Iterator argument, so fall back on a (slow) pure-Python solution
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
            keyed_max, maximum = keyed_min, minimum
        except StopIteration:
            raise ValueError('minmax argument is empty')
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




if __name__ == '__main__':
    import doctest
    failed, tried = doctest.testmod()
    if failed == 0:
        print("Successfully ran %d doctests." % tried)

