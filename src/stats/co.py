#!/usr/bin/env python3

##  Copyright (c) 2011 Steven D'Aprano.
##  See the file __init__.py for the licence terms for this software.

"""
Coroutine-based statistics functions.

"""

__all__ = [
    'sum', 'mean',
    'running_average', 'weighted_running_average', 'simple_moving_average',
    'pvariance1', 'variance1', 'pstdev1', 'stdev1',
    'corr1',
    'running_sum',
    ]


from builtins import sum as _sum
from . import StatsError
from .utils import coroutine, feed


# === Sums and averages ===

@coroutine
def sum(start=None, func=None):
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

    If optional argument func is given and is not None, it is applied to each
    value consumed:

    >>> rsum = sum(9, lambda x: x**2)
    >>> [rsum.send(n) for n in (1, 2, 3)]
    [10, 14, 23]

    """
    if start is not None:
        total = [start]
    else:
        total = []
    x = (yield None)
    if func is None:
        while True:
            add_partial(x, total)
            x = (yield _sum(total))
    else:
        while True:
            add_partial(func(x), total)
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

    that is, the average of the first item, the first two items, the first
    three items, the first four items, and so forth.

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
    with exponentially decreasing weights.

    >>> aver = weighted_running_average()

    The weighted running average consumes data
        a, b, c, d, ...

    and returns the values:
        a, (a+b)/2, ((a+b)/2 + c)/2, (((a+b)/2 + c)/2 + d)/2, ...

    This running average yields the average between the previous running
    average and the current data point. Given data [a, b, c, d, ...] it
    yields the values:
        a, (a+b)/2, ((a+b)/2 + c)/2, (((a+b)/2 + c)/2 + d)/2, ...

    The values yielded are weighted means where the weight on older points
    decreases exponentially.

    that is, the average of the first item, the first two items, the first
    three items, the first four items, and so forth.

    >>> list(weighted_running_average([40, 30, 50, 46, 39, 44]))
    [40, 35.0, 42.5, 44.25, 41.625, 42.8125]

    >>> aver = mean()
    >>> [aver.send(n) for n in (40, 30, 50, 46, 39, 44)]
    [40.0, 35.0, 40.0, 41.5, 41.0, 41.5]

    """
    ca = (yield None)
    x = (yield ca)
    while True:
        ca = (ca + x)/2
        x = (yield ca)


def simple_moving_average(data, window=3):
    """Iterate over data, yielding the simple moving average with a fixed
    window size (defaulting to three).

    >>> list(simple_moving_average([40, 30, 50, 46, 39, 44]))
    [40.0, 42.0, 45.0, 43.0]

    """
    it = iter(data)
    d = collections.deque(itertools.islice(it, window))
    if len(d) != window:
        raise StatsError('too few data points for given window size')
    s = sum(d)
    yield s/window
    for x in it:
        s += x - d.popleft()
        d.append(x)
        yield s/window


# === Measures of spread ===

def pvariance1(data):
    """pvariance1(data) -> population variance.

    Return an estimate of the population variance for data using one pass
    through the data. Use this when you can only afford a single path over
    the data -- if you can afford multiple passes, pvariance is likely to be
    more accurate.

    >>> pvariance1([0.25, 0.5, 1.25, 1.25,
    ...           1.75, 2.75, 3.5])  #doctest: +ELLIPSIS
    1.17602040816...

    If data represents a statistical sample rather than the entire
    population, then you should use variance1 instead.
    """
    n, s = _welford(data)
    if n < 1:
        raise StatsError('pvariance requires at least one data point')
    return s/n


def variance1(data):
    """variance1(data) -> sample variance.

    Return an estimate of the sample variance for data using a single pass.
    Use this when you can only afford a single path over the data -- if you
    can afford multiple passes, variance is likely to be more accurate.

    >>> variance1([0.25, 0.5, 1.25, 1.25,
    ...           1.75, 2.75, 3.5])  #doctest: +ELLIPSIS
    1.37202380952...

    If data represents the entire population rather than a statistical
    sample, then you should use pvariance1 instead.
    """
    n, s = _welford(data)
    if n < 2:
        raise StatsError('sample variance or standard deviation'
        ' requires at least two data points')
    return s/(n-1)


def _welford(data):
    """Welford's method of calculating the running variance.

    This calculates the second moment M2 = sum( (x-m)**2 ) where m=mean of x.
    Returns (n, M2) where n = number of items.
    """
    # Note: for better results, use this on the residues (x - m) instead of x,
    # where m equals the mean of the data... except that would require two
    # passes, which we're trying to avoid.
    data = iter(data)
    n = 0
    M2 = 0.0  # Current sum of powers of differences from the mean.
    try:
        m = next(data)  # Current estimate of the mean.
        n = 1
    except StopIteration:
        pass
    else:
        for n, x in enumerate(data, 2):
            delta = x - m
            m += delta/n
            M2 += delta*(x - m)  # m here is the new, updated mean.
    assert M2 >= 0.0
    return (n, M2)


def pstdev1(data):
    """pstdev1(data) -> population standard deviation.

    Return an estimate of the population standard deviation for data using
    a single pass. Use this when you can only afford a single path over the
    data -- if you can afford multiple passes, pstdev is likely to be more
    accurate.

    >>> pstdev1([1.5, 2.5, 2.5, 2.75, 3.25, 4.75])  #doctest: +ELLIPSIS
    0.986893273527...

    If data is a statistical sample rather than the entire population, you
    should use stdev1 instead.
    """
    return math.sqrt(pvariance1(data))


def stdev1(data):
    """stdev1(data) -> sample standard deviation.

    Return an estimate of the sample standard deviation for data using
    a single pass. Use this when you can only afford a single path over the
    data -- if you can afford multiple passes, stdev is likely to be more
    accurate.

    >>> stdev1([1.5, 2.5, 2.5, 2.75, 3.25, 4.75])  #doctest: +ELLIPSIS
    1.08108741552...

    If data represents the entire population rather than a statistical
    sample, then use pstdev1 instead.
    """
    return math.sqrt(variance1(data))



# === Other moments of the data ===

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

def corr1(xydata):
    """corr1(xydata) -> float

    Calculate an estimate of the Pearson's correlation coefficient with
    a single pass over iterable xydata. See also the function corr which may
    be more accurate but requires multiple passes over the data.

    >>> data = zip([0, 5, 4, 9, 8, 4], [1, 2, 4, 8, 6, 3])
    >>> corr1(data)  #doctest: +ELLIPSIS
    0.903737838893...

    xydata must be an iterable of (x, y) points. Raises StatsError if there
    are fewer than two points, or if either of the estimated x and y variances
    are zero.
    """
    xydata = iter(xydata)
    sum_sq_x = 0
    sum_sq_y = 0
    sum_coproduct = 0
    try:
        mean_x, mean_y = next(xydata)
    except StopIteration:
        i = 0
    else:
        i = 1
        for i,(x,y) in zip(itertools.count(2), xydata):
            sweep = (i-1)/i
            delta_x = x - mean_x
            delta_y = y - mean_y
            sum_sq_x += sweep*delta_x**2
            sum_sq_y += sweep*(delta_y**2)
            sum_coproduct += sweep*(delta_x*delta_y)
            mean_x += delta_x/i
            mean_y += delta_y/i
    if i < 2:
        raise StatsError('correlation coefficient requires two or more items')
    pop_sd_x = math.sqrt(sum_sq_x)
    pop_sd_y = math.sqrt(sum_sq_y)
    if pop_sd_x == 0.0:
        raise StatsError('calculated x variance is zero')
    if pop_sd_y == 0.0:
        raise StatsError('calculated y variance is zero')
    r = sum_coproduct/(pop_sd_x*pop_sd_y)
    # r can sometimes exceed the limits -1, 1 by up to 2**-51. We accept
    # that without comment.
    excess = max(abs(r) - 1.0, 0.0)
    if 0 < excess <= 2**-51:
        r = math.copysign(1, r)
    assert -1.0 <= r <= 1.0, "expected -1.0 <= r <= 1.0 but got r = %r" % r
    return r


if __name__ == '__main__':
    import doctest
    doctest.testmod()

