#!/usr/bin/env python3

##  Copyright (c) 2011 Steven D'Aprano.
##  See the file __init__.py for the licence terms for this software.

"""
Coroutine-based statistics functions
------------------------------------

These functions provide a consumer model for calculating statistics using
coroutines. For example, to calculate a running sum, you use the send()
method to pass values into the consumer, and get the running sum back:

>>> import stats.co
>>> running_sum = stats.co.sum()
>>> running_sum.send(42)
42
>>> running_sum.send(23)
65

Each time you send a value into the consumer, the running total is updated
and returned. A consumer version of the mean is also provided:

>>> rmean = stats.co.mean()
>>> rmean.send(1)
1.0
>>> rmean.send(5)
3.0

Consumer versions of variance and stdev functions (both population and
sample) and Pearson's correlation coefficient are also provided. They may
be particularly useful when you have a very large data stream and don't
want to make multiple passes over the data.

To convert a consumer into a producer, use the helper function `feed`. This
takes two arguments, a consumer such as a coroutine and an iterable data
source, and returns a generator object that yields the output of sending
data into the consumer:

>>> rsum = stats.co.sum()  # Create a consumer of data.
>>> it = stats.co.feed(rsum, [1, 4, 7])  # Turn it into a producer.
>>> next(it)
1
>>> next(it)
5
>>> next(it)
12

The weighted_running_average consumer accepts data points and returns the
running average of the current data point and the previous average:

>>> aver = stats.co.weighted_running_average()
>>> aver.send(3)
3
>>> aver.send(5)
4.0
>>> aver.send(2)
3.0
>>> it = stats.co.feed(aver, [1, 4, 7])
>>> list(it)
[2.0, 3.0, 5.0]

"""

__all__ = [
    'feed', 'sum', 'mean', 'weighted_running_average',
    'pvariance', 'variance', 'pstdev', 'stdev', 'corr',
    ]


import collections
import itertools
import math

from builtins import sum as _sum
from stats.utils import StatsError, coroutine, add_partial


# === Utilities and helpers ===

def feed(consumer, iterable):
    """feed(consumer, iterable) -> yield items

    Helper function to convert a consumer coroutine into a producer.
    feed() returns a generator that yields items from the given coroutine
    and iterator.

    >>> def counter():              # Consumer that counts items sent in.
    ...     c = 0
    ...     _ = (yield None)
    ...     while True:
    ...             c += 1
    ...             _ = (yield c)
    ... 
    >>> cr = counter()
    >>> cr.send(None)  # Prime the coroutine.
    >>> list(feed(cr, ["spam", "ham", "eggs"]))  # Send many values.
    [1, 2, 3]
    >>> cr.send("spam and eggs")  # Manually sending still works.
    4

    """
    for obj in iterable:
        yield consumer.send(obj)


# === Sums and averages ===

@coroutine
def sum(start=None):
    """Running sum co-routine.

    With no arguments, sum() consumes values and returns the running sum:

    >>> rsum = sum()
    >>> rsum.send(1)
    1
    >>> [rsum.send(n) for n in (2, 3, 4)]
    [3, 6, 10]

    If optional argument start is given and is not None, it is used as the
    initial value for the running sum:

    >>> rsum = sum(9)
    >>> [rsum.send(n) for n in (1, 2, 3)]
    [10, 12, 15]

    """
    if start is not None:
        total = [start]
    else:
        total = []
    x = (yield None)
    while True:
        add_partial(x, total)
        x = (yield _sum(total))


@coroutine
def mean():
    """Running mean co-routine.

    mean() consumes values and returns the running average:

    >>> aver = mean()
    >>> aver.send(1)
    1.0
    >>> [aver.send(n) for n in (2, 3, 4)]
    [1.5, 2.0, 2.5]

    The running average, or cumulative moving average, consumes data
        a, b, c, d, ...

    and returns the values:
        a/1, (a+b)/2, (a+b+c)/3, (a+b+c+d)/4, ...

    >>> aver = mean()
    >>> [aver.send(n) for n in (40, 30, 50, 46, 39, 44)]
    [40.0, 35.0, 40.0, 41.5, 41.0, 41.5]

    """
    total = []
    n = 0
    x = (yield None)
    while True:
        add_partial(x, total)
        n += 1
        x = (yield _sum(total)/n)


@coroutine
def weighted_running_average():
    """Weighted running average co-routine.

    weighted_running_average() consumes values and returns a running average
    with exponentially decreasing weights. The first value returned is the
    first data point consumed; after that, each value is the average between
    the previous average and the most-recent data point:

    >>> aver = weighted_running_average()
    >>> aver.send(5)
    5
    >>> aver.send(1)  # average of 5 and 1
    3.0
    >>> aver.send(2)  # average of 3 and 2
    2.5

    The weighted running average consumes data
        a, b, c, d, ...

    and returns the values:
        a, (a+b)/2, ((a+b)/2 + c)/2, (((a+b)/2 + c)/2 + d)/2, ...

    The values returned are weighted means, where the weight on older points
    decreases exponentially.

    >>> aver = weighted_running_average()
    >>> [aver.send(n) for n in (40, 30, 50, 46, 39, 44)]
    [40, 35.0, 42.5, 44.25, 41.625, 42.8125]

    """
    ca = (yield None)
    x = (yield ca)
    while True:
        ca = (ca + x)/2
        x = (yield ca)


# === Measures of spread ===

@coroutine
def _welford():
    """Welford's method of calculating the running variance.

    Consume values and return running estimates of (n, M2) where:
        n = number of data points seen so far
        M2 = the second moment about the mean
           = sum( (x-m)**2 ) where m = mean of the x seen so far.

    """
    # Note: for better results, use this on the residues (x - m) instead
    # of the raw x values, where m equals the mean of the data.
    M2_partials = []
    x = (yield None)
    m = x  # First estimate of the mean is the first value.
    n = 1
    while True:
        delta = x - m
        m += delta/n  # Update the mean.
        add_partial(delta*(x-m), M2_partials)  # Update the second moment.
        M2 = _sum(M2_partials)
        assert M2 >= 0.0
        x = (yield (n, M2))
        n += 1


@coroutine
def pvariance():
    """Running population variance co-routine.

    pvariance() consumes values and returns the population variance of the
    data points seen so far:

    >>> data = [0.25, 0.5, 1.25, 1.25, 1.75, 2.75, 3.5]
    >>> rvar = pvariance()
    >>> for x in data:
    ...     print(rvar.send(x))
    ...     #doctest: +ELLIPSIS
    0.0
    0.015625
    0.18055555555...
    0.19921875
    0.3
    0.67534722222...
    1.17602040816...

    This may be especially useful when you can only afford a single pass
    through the data.

    If your data represents a statistical sample rather than the entire
    population, then you should use variance instead.
    """
    cr = _welford()
    x = (yield None)
    n, M2 = cr.send(x)
    while True:
        n, M2 = cr.send((yield M2/n))


@coroutine
def variance():
    """Running sample variance co-routine.

    variance() consumes values and returns the sample variance of the
    data points seen so far. Note that the sample variance is never defined
    for a single data point, and a float NAN will always be returned.

    >>> data = [0.25, 0.5, 1.25, 1.25, 1.75, 2.75, 3.5]
    >>> rvar = variance()
    >>> for x in data:
    ...     print(rvar.send(x))
    ...     #doctest: +ELLIPSIS
    nan
    0.03125
    0.27083333333...
    0.265625
    0.375
    0.81041666666...
    1.37202380952...

    This may be especially useful when you can only afford a single pass
    through the data.

    If your data represents the entire population rather than a statistical
    sample, then you should use pvariance instead.
    """
    cr = _welford()
    x = (yield None)
    n, M2 = cr.send(x)
    assert n == 1 and M2 == 0.0
    x = (yield float('nan'))
    n, M2 = cr.send(x)
    while True:
        n, M2 = cr.send((yield M2/(n-1)))


@coroutine
def pstdev():
    """Running population standard deviation co-routine.

    pstdev() consumes values and returns the population standard deviation
    of the data points seen so far:

    >>> data = [1.75, 0.25, 1.25, 3.5, 2.75, 1.25, 0.5]
    >>> rsd = pstdev()
    >>> for x in data:
    ...     print(rsd.send(x))
    ...     #doctest: +ELLIPSIS
    0.0
    0.75
    0.62360956446...
    1.17759023009...
    1.13578166916...
    1.0647443616
    1.08444474648...

    This may be especially useful when you can only afford a single pass
    through the data.

    If your data represents a statistical sample rather than the entire
    population, then you should use stdev instead.
    """
    var = pvariance()
    x = (yield None)
    x = var.send(x)
    while True:
        x = var.send((yield math.sqrt(x)))


@coroutine
def stdev():
    """Running sample standard deviation co-routine.

    stdev() consumes values and returns the sample standard deviation
    of the data points seen so far. Note that the sample standard deviation
    is never defined for a single data point, and a float NAN will always
    be returned.

    >>> data = [1.75, 0.25, 1.25, 3.5, 2.75, 1.25, 0.5]
    >>> rsd = stdev()
    >>> for x in data:
    ...     print(rsd.send(x))
    ...     #doctest: +ELLIPSIS
    nan
    1.06066017178...
    0.76376261582...
    1.35976407267...
    1.26984250992...
    1.16636900965...
    1.17133420061...

    This may be especially useful when you can only afford a single pass
    through the data.

    If your data represents the entire population rather than a statistical
    sample, then you should use pstdev instead.
    """
    var = variance()
    x = (yield None)
    x = var.send(x)
    while True:
        x = var.send((yield math.sqrt(x)))



# === Other moments of the data ===

# FIX ME
def _terriberry(data):
    """Terriberry's algorithm for a single pass estimate of skew and kurtosis.

    This calculates the second, third and fourth moments
        M2 = sum( (x-m)**2 )
        M3 = sum( (x-m)**3 )
        M4 = sum( (x-m)**4 )
    where m = mean of x.

    Returns (n, M2, M3, M4) where n = number of items.
    """
    n = m = M2 = M3 = M4 = 0
    for n, x in enumerate(data, 1):
        delta = x - m
        delta_n = delta/n
        delta_n2 = delta_n*delta_n
        term = delta*delta_n*(n-1)
        m += delta_n
        M4 += term*delta_n2*(n*n - 3*n + 3) + 6*delta_n2*M2 - 4*delta_n*M3
        M3 += term*delta_n*(n-2) - 3*delta_n*M2
        M2 += term
    return (n, M2, M3, M4)
    # skewness = sqrt(n)*M3 / sqrt(M2**3)
    # kurtosis = (n*M4) / (M2*M2) - 3


# === Multivariate functions ===

def _calc_r(sumsqx, sumsqy, sumco):
    """Helper function to calculate r."""
    sx = math.sqrt(sumsqx)
    sy = math.sqrt(sumsqy)
    den = sx*sy
    if den == 0.0:
       return float('nan')
    r = sumco/den
    # -1 <= r <= +1 should hold, but due to rounding errors sometimes the
    # absolute value of r can exceed 1 by up to 2**-51. We accept this
    # without comment.
    excess = max(abs(r) - 1.0, 0.0)
    if 0 < excess <= 2**-51:
        r = math.copysign(1, r)
    assert -1.0 <= r <= 1.0, "expected -1.0 <= r <= 1.0 but got r = %r" % r
    return r


@coroutine
def corr():
    """Running Pearson's correlation coefficient coroutine.

    corr() consumes (x,y) pairs and returns r, the Pearson's correlation
    coefficient, of the data points seen so far. Note that r requires at
    least two pairs of data to be defined, and consequently the first value
    returned will be a float NAN.

    >>> xdata = [0, 5, 4, 9, 8, 4, 3]
    >>> ydata = [1, 2, 4, 8, 6, 3, 4]
    >>> rr = corr()
    >>> for x,y in zip(xdata, ydata):
    ...     print(rr.send((x, y)))
    ...
    nan
    1.0
    0.618589574132
    0.888359981681
    0.901527628267
    0.903737838894
    0.875341049362

    This may be especially useful when you can only afford a single pass
    through the data.

    r is always between -1 and 1 inclusive. If the estimated variance of
    either the x or the y data points is zero (e.g. if they are constant),
    or if their product underflows to zero, a float NAN will be returned.
    """
    sumsqx = 0  # sum of the squares of the x values
    sumsqy = 0  # sum of the squares of the y values
    sumco = 0  # sum of the co-product x*y
    i = 1
    x,y = (yield None)
    mx = x  # First estimate of the means are the first values.
    my = y
    while True:
        sweep = (i-1)/i
        dx = x - mx
        dy = y - my
        sumsqx += sweep*dx**2
        sumsqy += sweep*(dy**2)
        sumco += sweep*(dx*dy)
        mx += dx/i  # Update the means.
        my += dy/i
        r = _calc_r(sumsqx, sumsqy, sumco)
        x,y = (yield r)
        i += 1

