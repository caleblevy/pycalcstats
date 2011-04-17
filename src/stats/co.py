#!/usr/bin/env python3

##  Copyright (c) 2011 Steven D'Aprano.
##  See the file __init__.py for the licence terms for this software.

"""
The ``stats.co`` module provides eight coroutine based statistics functions:

    Function        Description
    ==============  ==========================================
    corr            Correlation coefficient of (X, Y) data.
    ewma            Exponentially weighted moving average.
    mean            Running arithmetic mean (average).
    pstdev          Population standard deviation of data.
    pvariance       Population variance of data.
    stdev           Sample standard deviation of data.
    sum             Running sum of data.
    variance        Sample variance of data (bias-corrected).

The function ``stats.co.sum`` is an alias to ``stats.running_sum``.

The module also includes one public utility function:

    Name                Description
    ==================  =============================================
    feed                Convert coroutines into iterators.



Consumers
---------

The functions in this module use a consumer model based on coroutines.
Instead of passing the entire data set to the function as a single argument,
you pass each data point to the function one at a time, and the result is
incrementally calculated. This is ideal for calculating running statistics
or for dealing with data sets that is produced lazily, particularly if you
have a very large data stream and don't want to, or can't, make multiple
passes over the data.

For example, to calculate a running sum, you use the send() method to pass
values into the consumer, and get the running sum back:

>>> import stats.co
>>> running_sum = stats.co.sum()
>>> running_sum.send(42)
42
>>> running_sum.send(23)
65

Each time you send a value into the consumer, the running total is updated
and returned. Similarly, to calculate a running mean:

>>> running_mean = stats.co.mean()
>>> running_mean.send(2)
2.0
>>> running_mean.send(3)
2.5
>>> running_mean.send(4)
3.0



Producers
---------

Consumers take their input lazily; producers give up their output lazily.
To convert a consumer into a producer, use the helper function ``feed``.
This takes two arguments, a coroutine (the consumer) and an iterable data
source, and returns a iterator that yields the output of sending data into
the consumer. For example:

>>> running_sum = stats.co.sum()  # Create a consumer of data.
>>> it = stats.co.feed(running_sum, [1, 4, 7])  # Turn it into a producer.
>>> next(it)
1
>>> next(it)
5
>>> next(it)
12

"""

__all__ = [
    'corr', 'ewma', 'feed', 'mean', 'pstdev', 'pvariance', 'stdev',
    'sum', 'variance',
    ]


import collections
import itertools
import math

import stats

from builtins import sum as _sum



# === Utilities and helpers ===

def feed(consumer, iterable):
    """feed(consumer, iterable) -> yield items

    Helper function to convert a consumer coroutine into a producer.
    feed() returns a generator that yields items from the given coroutine
    and iterator.

    >>> def counter():  # Consumer that counts items sent in.
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

from stats import running_sum as sum

@stats.coroutine
def mean():
    """Running mean co-routine.

    mean() consumes values and returns the running average:

    >>> aver = mean()
    >>> aver.send(1)
    1.0
    >>> [aver.send(n) for n in (2, 3, 4)]
    [1.5, 2.0, 2.5]

    The running average, also known as the cumulative moving average,
    consumes data:

        a, b, c, d, ...

    and returns the values:

        a, (a+b)/2, (a+b+c)/3, (a+b+c+d)/4, ...

    >>> aver = mean()
    >>> [aver.send(n) for n in (40, 30, 50, 46, 39, 44)]
    [40.0, 35.0, 40.0, 41.5, 41.0, 41.5]

    """
    n = 0
    running_sum = sum()
    x = (yield None)
    while True:
        total = running_sum.send(x)
        n += 1
        x = (yield total/n)


@stats.coroutine
def ewma(alpha=0.5):
    """Exponentially weighted moving average (EWMA).

    Coroutine returning a moving average with exponentially decreasing
    weights. The first value returned is the first data point consumed.

    Optional parameter ``alpha`` controls the degree by which the weights
    decrease. It must be a number, and should be between 0 and 1 exclusive.
    (Values of ``alpha`` outside of this range are not prohibited, but the
    results may not be physically or statistically meaningful.) Each
    moving average after the initial point is given by:

        alpha * current data point + (1 - alpha) * previous average

    >>> aver = ewma(0.25)
    >>> aver.send(3)
    3
    >>> aver.send(5)
    3.5
    >>> aver.send(2)
    3.125
    >>> aver.send(4)
    3.34375

    By default ``alpha`` is one half, which is equivalent to averaging each
    value (after the first) with the previous moving average:

    >>> aver = ewma()
    >>> aver.send(5)
    5
    >>> aver.send(1)  # average of 5 and 1
    3.0
    >>> aver.send(2)  # average of 3 and 2
    2.5
    >>> it = feed(aver, [4.5, 2.5, 4.0])
    >>> list(it)
    [3.5, 3.0, 3.5]

    >>> aver = ewma()
    >>> [aver.send(n) for n in (40, 30, 50, 46, 39, 44)]
    [40, 35.0, 42.5, 44.25, 41.625, 42.8125]

    """
    if not stats._is_numeric(alpha):
        raise stats.StatsError('alpha must be a number')
    complement_alpha = 1 - alpha
    moving_average = (yield None)
    x = (yield moving_average)
    while True:
        try:
            moving_average = alpha*x + complement_alpha*moving_average
        except TypeError:
            if not stats._is_numeric(x):
                raise
            # Downgrade to floats and try again.
            x = float(x)
            moving_average = float(moving_average)
            alpha = float(alpha)
            complement_alpha = 1 - alpha
            continue
        x = (yield moving_average)


# === Measures of spread ===

@stats.coroutine
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
        stats.add_partial(delta*(x-m), M2_partials)  # Update the 2nd moment.
        M2 = _sum(M2_partials)
        assert M2 >= 0.0
        x = (yield (n, M2))
        n += 1


@stats.coroutine
def pvariance():
    """Running population variance co-routine.

    ``pvariance`` consumes values and returns the variance with N degrees of
    freedom of the data points seen so far:

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

    """
    cr = _welford()
    x = (yield None)
    n, M2 = cr.send(x)
    while True:
        n, M2 = cr.send((yield M2/n))


@stats.coroutine
def variance():
    """Running sample variance co-routine.

    ``variance`` consumes values and returns the variance with N-1 degrees
    of freedom of the data points seen so far.

        WARNING: The sample variance with N-1 degrees of freedom
        is not defined for a single data point. The first result
        given by ``variance`` will always be a NAN.

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

    """
    cr = _welford()
    x = (yield None)
    n, M2 = cr.send(x)
    assert n == 1 and M2 == 0
    x = (yield float('nan'))
    n, M2 = cr.send(x)
    while True:
        n, M2 = cr.send((yield M2/(n-1)))


@stats.coroutine
def pstdev():
    """Running population standard deviation co-routine.

    ``pstdev`` consumes values and returns the standard deviation with N
    degrees of freedom for the data points seen so far:

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

    """
    var = pvariance()
    x = (yield None)
    x = var.send(x)
    while True:
        x = var.send((yield math.sqrt(x)))


@stats.coroutine
def stdev():
    """Running sample standard deviation co-routine.

    ``stdev`` consumes values and returns the standard deviation with N-1
    degrees of freedom for the the data points seen so far.

        WARNING: The sample standard deviation with N-1 degrees of
        freedom is not defined for a single data point. The first
        result given by ``stdev`` will always be a NAN.

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

    This is (currently) completely untested and unsupported.

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
    # absolute value of r can exceed 1 by up to 2**-50. We accept this
    # without comment.
    excess = max(abs(r) - 1.0, 0.0)
    if 0 < excess <= 2**-50:
        r = math.copysign(1, r)
    assert -1.0 <= r <= 1.0, "expected -1.0 <= r <= 1.0 but got r = %r" % r
    return r


@stats.coroutine
def corr():
    """Running Pearson's correlation coefficient coroutine ``r``.

    ``corr`` consumes (X,Y) pairs and returns ``r``, the sample Pearson's
    correlation coefficient, of the data points seen so far.

        WARNING: Pearson's correlation coefficient (with N-1 degrees
        of freedom) is not defined for a single data point. The first
        result given by ``corr`` will always be a NAN.

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

    The correlation coefficient ``r`` is a number between -1 and 1 inclusive.
    It gives a measure of the strength of *linear* relationship between the
    X and Y coordinates of the data. A correlation of 1 implies a perfect
    linear relationship between the X and Y values, where increasing X gives
    increasing Y, and vice versa. A correlation of -1 similarly implies a
    perfect linear relationship where increasing X gives decreasing Y,
    and vice versa (a perfect anti-correlation). A correlation of zero
    implies no linear relationship between the X and Y coordinates.

    ``r`` is always between -1 and 1 inclusive, unless it is a NAN.
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

