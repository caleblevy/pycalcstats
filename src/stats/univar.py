#!/usr/bin/env python3

##  Copyright (c) 2011 Steven D'Aprano.
##  See the file __init__.py for the licence terms for this software.

"""
Univariate statistics.


This module provides the following univariate statistics functions:

    Function            Description
    ==================  ===============================================
    average_deviation   Average deviation from a central location.
    circular_mean       Mean (average) of circular quantities.
    geometric_mean*     Mean of exponential growth rates.
    harmonic_mean*      Mean of rates or speeds.
    kurtosis            Measure of shape of the data.
    mode                Most frequent value.
    moving_average      Simple moving average iterator.
    pearson_skewness    Measure of symmetry of the data.
    quadratic_mean*     Root-mean-square average.
    skewness            Measure of the symmetry of the data
    sterrkurtosis       Standard error of the kurtosis.
    sterrmean           Standard error of the mean.
    sterrskewness       Standard error of the skewness.

Functions marked with * can operate on columnar data. See the documentation
for the ``stats`` module, or the indiviual function, for further details.

"""

__all__ = [
    'average_deviation', 'circular_mean', 'geometric_mean', 'harmonic_mean',
    'kurtosis', 'mode', 'moving_average', 'pearson_skewness',
    'quadratic_mean', 'skewness', 'sterrkurtosis', 'sterrmean',
    'sterrskewness',
    ]

import math
import operator
import functools
import itertools
import collections

import stats
import stats.utils


# Utility functions
# -----------------

def make_freq_table(data):
    """Return a frequency table from the elements of data.

    >>> d = make_freq_table([1.5, 2.5, 1.5, 0.5])
    >>> sorted(d.items())
    [(0.5, 1), (1.5, 2), (2.5, 1)]

    """
    D = {}
    for element in data:
        D[element] = D.get(element, 0) + 1
    return D  #collections.Counter(data)


def _divide(num, den):
    """Return num/div without raising unnecessary exceptions.

    >>> _divide(1, 0)
    inf
    >>> from decimal import Decimal
    >>> _divide(Decimal(0), 0)
    Decimal('NaN')

    """
    # Support Decimal, but only if necessary. Avoid importing such a
    # heavyweight module if not needed.
    if 'decimal' in (type(num).__module__, type(den).__module__):
        import decimal
        with decimal.localcontext() as ctx:
            ctx.traps[decimal.DivisionByZero] = 0
            ctx.traps[decimal.InvalidOperation] = 0
            return num/den
    # Support non-Decimal values.
    try:
        return num/den
    except ZeroDivisionError:
        if num:
            result = math.copysign(float('inf'), den)
            if num < 0:
                result = -result
        else:
            result = float('nan')
    for x in (num, den):
        try:
            return x.from_float(result)
        except (AttributeError, ValueError, TypeError):
            pass
    return result


# Measures of central tendency (means and averages)
# -------------------------------------------------

def harmonic_mean(data):
    """harmonic_mean(iterable_of_numbers) -> harmonic mean of numbers
    harmonic_mean(iterable_of_rows) -> harmonic means of columns

    Return the harmonic mean of the given numbers or columns.

    The harmonic mean, or subcontrary mean, is the reciprocal of the
    arithmetic mean of the reciprocals of the data. It is a type of average
    best used for averaging rates or speeds.

    >>> harmonic_mean([0.25, 0.5, 1.0, 1.0])
    0.5

    If data includes one or more zero values, the result will be zero if the
    zeroes are all the same sign, or an NAN if they are of opposite signs.

    When passed an iterable of sequences, each inner sequence represents a
    row of data, and ``harmonic_mean`` operates on each column. All rows
    must have the same number of columns, or ValueError is raised.

    >>> data = [[0, 1, 2, 4],
    ...         [1, 2, 4, 8],
    ...         [2, 4, 8, 8]]
    ...
    >>> harmonic_mean(data)  #doctest: +ELLIPSIS
    [0.0, 1.71428..., 3.42857..., 6.0]

    """
    # FIXME harmonic_mean([x]) should equal x exactly, but due to rounding
    # errors in the 1/(1/x) round trip, sometimes it doesn't.
    invert = functools.partial(_divide, 1)
    count, total = stats._len_sum(data, invert)
    if not count:
        raise stats.StatsError('harmonic mean of empty sequence is not defined')
    f = functools.partial(_divide, count)
    return stats._vsmap(f, total)


def geometric_mean(data):
    """Return the sample geometric mean of a sequence of non-negative numbers.

    >>> geometric_mean([1.0, 2.0, 6.125, 12.25])
    3.5

    The geometric mean of N items is the Nth root of the product of the
    items. It is best suited for averaging exponential growth rates.

    """
    # Calculate the length and product of data.
    def safe_mul(a, b):
        x = a*b
        if x < 0: return float('nan')
        return x
    scalar_multiply = functools.partial(functools.reduce, safe_mul)
    count, total = stats._generalised_reduce(
                    scalar_multiply, stats.running_product, data)
    if not count:
        raise stats.StatsError(
        'geometric mean of empty sequence is not defined')
    return pow(total, 1.0/count)


def quadratic_mean(data):
    """quadratic_mean(iterable_of_numbers) -> quadratic mean of numbers
    quadratic_mean(iterable_of_rows) -> quadratic means of columns

    Return the quadratic mean of the given numbers or columns.

    >>> quadratic_mean([2, 2, 4, 5])
    3.5

    The quadratic mean, or RMS (Root Mean Square), is the square root of the
    arithmetic mean of the squares of the data. It is a type of average
    best used to get an average absolute magnitude when quantities vary from
    positive to negative:

    >>> quadratic_mean([-3, -2, 0, 2, 3])
    2.280350850198276

    When passed an iterable of sequences, each inner sequence represents a
    row of data, and ``quadratic_mean`` operates on each column. All rows
    must have the same number of columns, or ValueError is raised.

    >>> data = [[0, 1, 2, 4],
    ...         [1, 2, 4, 6],
    ...         [2, 4, 6, 6]]
    ...
    >>> quadratic_mean(data)  #doctest: +ELLIPSIS
    [1.29099..., 2.64575..., 4.3204..., 5.41602...]

    """
    count, total = stats._len_sum(data, lambda x: x*x)
    if not count:
        raise stats.StatsError('quadratic mean of empty sequence is not defined')
    return stats._vsmap(lambda x: math.sqrt(x/count), total)


def mode(data, window=None):
    """Returns the most common element of a sequence of numbers.

    The mode is commonly used as an average. It is the "most typical"
    value of a distribution or data set.

    >>> mode([5, 7, 2, 3, 2, 2, 1, 3])
    2

    For discrete data, pass ``window=None`` (the default) to return the
    element with the largest frequency. If there is no such element, or
    it is not unique, ``StatsError`` is raised.

    For continuous data, the mode is estimated using the technique described
    as "estimating the rate of an inhomogeneous Poisson process by jth
    waiting times" (Numerical Recipes in Pascal, Press et. al.) Choose a
    positive integer as the "window size". A smaller window size gives better
    resolution and a chance at finding a high but narrow peak, but is also
    more likely to mistake a chance fluctuation in the data as the mode.

    ``window`` should be as large as you can tolerate. If it is less than 1,
    ValueError is raised. If it is 2 or 3, a warning is raised.
    """
    if window is not None:
        if window < 1:
            raise ValueError('window size must be strictly positive and'
            ' should be at least three')
        if window < 3:
            import warnings
            warnings.warn('window size is recommended to be at least three')
        raise NotImplementedError
    assert window is None
    L = sorted(
        [(count, value) for (value, count) in
         make_freq_table(data).items()],
         reverse=True)
    if len(L) == 0:
        raise stats.StatsError('no mode is defined for empty iterables')
    # Test if there are more than one modes.
    if len(L) > 1 and L[0][0] == L[1][0]:
        raise stats.StatsError('no distinct mode')
    return L[0][1]


def moving_average(data, window=3):
    """Iterate over data, yielding the simple moving average with a fixed
    window size.

    With a window size of N (defaulting to three), the simple moving average
    yields the average of items data[0:N], data[1:N+1], data[2:N+2], ...

    >>> list(moving_average([40, 30, 50, 46, 39, 44]))
    [40.0, 42.0, 45.0, 43.0]

    """
    it = iter(data)
    d = collections.deque(itertools.islice(it, window))
    if len(d) != window:
        raise ValueError('too few data points for given window size')
    s = sum(d)
    yield s/window
    for x in it:
        s += x - d.popleft()
        d.append(x)
        yield s/window


# Measures of spread (dispersion or variability)
# ----------------------------------------------

def average_deviation(data, m=None):
    """average_deviation(data [, m]) -> average absolute deviation of data.

    Returns the average deviation of the sample data from the population
    centre ``m`` (usually the mean, or the median). If you know the
    population mean or median, pass it as the second element:

    >>> data = [2.0, 2.25, 2.5, 2.5, 3.25]  # A sample from a population
    >>> mu = 2.75                           # with a known mean.
    >>> average_deviation(data, mu)
    0.45

    If you don't know the centre location, you can estimate it by passing
    the sample mean or median instead. If ``m`` is not None, or not given,
    the sample mean is calculated from the data and used:

    >>> average_deviation(data)
    0.3

    """
    if m is None:
        if not isinstance(data, (list, tuple)):
            data = list(data)
        m = stats.mean(data)
    n, total = stats._len_sum(data, lambda x: abs(x-m))
    if n < 1:
        raise stats.StatsError(
        'average deviation requires at least 1 data point')
    return total/n


# Other moments of the data
# -------------------------

def pearson_skewness(mean, mode, stdev):
    """Return the Pearson Mode Skewness from the mean, mode and standard
    deviation of a data set.

    >>> pearson_skewness(2.5, 2.25, 2.5)
    0.1
    >>> pearson_skewness(2.5, 5.75, 0.5)
    -6.5

    """
    if stdev > 0:
        return (mean-mode)/stdev
    elif stdev == 0:
        return float('nan') if mode == mean else float('inf')
    else:
        raise stats.StatsError("standard deviation cannot be negative")


def skewness(data, m=None, s=None):
    """skewness(data [,m [,s]]) -> sample skewness of data.

    Returns a biased estimate of the degree to which the data is skewed to
    the left or the right of the mean.

    >>> skewness([1.25, 1.5, 1.5, 1.75, 1.75, 2.5, 2.75, 4.5])
    ... #doctest: +ELLIPSIS
    1.12521290135...

    If you know one or both of the population mean and standard deviation,
    or estimates of them, then you can pass the mean as optional argument m
    and the standard deviation as s.

    >>> skewness([1.25, 1.5, 1.5, 1.75, 1.75, 2.5, 2.75, 4.5], m=2.25)
    ... #doctest: +ELLIPSIS
    0.965559535600599...

    The reliablity of the result as an estimate for the true skewness depends
    on the estimated mean and standard deviation. If m or s are not given, or
    are None, they are estimated from the data.

    A negative skewness indicates that the distribution's left-hand tail is
    longer than the tail on the right-hand side, and that the majority of
    the values (including the median) are to the right of the mean. A
    positive skew indicates that the right-hand tail is longer, and that the
    majority of values are to the left of the mean. A zero skew indicates
    that the values are evenly distributed around the mean, often but not
    necessarily implying the distribution is symmetric.

        :: CAUTION ::
        As a rule of thumb, a non-zero value for skewness should only be
        treated as meaningful if its absolute value is larger than
        approximately twice its standard error. See stderrskewness.

    """
    if m is None or s is None:
        if not isinstance(data, (list, tuple)):
            data = list(data)
        if m is None: m = stats.mean(data)
        if s is None: s = stats.stdev(data, m)
    n, total = stats._len_sum(data, lambda x: ((x-m)/s)**3)
    return total/n


def kurtosis(data, m=None, s=None):
    """kurtosis(data [,m [,s]]) -> sample excess kurtosis of data.

    Returns a biased estimate of the excess kurtosis of the data, relative
    to the kurtosis of the normal distribution. To convert to kurtosis proper,
    add 3 to the result.

    >>> kurtosis([1.25, 1.5, 1.5, 1.75, 1.75, 2.5, 2.75, 4.5])
    ... #doctest: +ELLIPSIS
    -0.1063790369...

    If you know one or both of the population mean and standard deviation,
    or estimates of them, then you can pass the mean as optional argument m
    and the standard deviation as s.

    >>> kurtosis([1.25, 1.5, 1.5, 1.75, 1.75, 2.5, 2.75, 4.5], m=2.25)
    ... #doctest: +ELLIPSIS
    -0.37265014648437...

    The reliablity of the result as an estimate for the kurtosis depends on
    the estimated mean and standard deviation given. If m or s are not given,
    or are None, they are estimated from the data.

    The kurtosis of a population is a measure of the peakedness and weight
    of the tails. The normal distribution has kurtosis of zero; positive
    kurtosis generally has heavier tails and a sharper peak than normal;
    negative kurtosis generally has lighter tails and a flatter peak.

    There is no upper limit for kurtosis, and a lower limit of -2. Higher
    kurtosis means more of the variance is the result of infrequent extreme
    deviations, as opposed to frequent modestly sized deviations.

        :: CAUTION ::
        As a rule of thumb, a non-zero value for kurtosis should only
        be treated as meaningful if its absolute value is larger than
        approximately twice its standard error. See stderrkurtosis.

    """
    if m is None or s is None:
        if not isinstance(data, (list, tuple)):
            data = list(data)
        if m is None: m = stats.mean(data)
        if s is None: s = stats.stdev(data, m)
    n, total = stats._len_sum(data, lambda x: ((x-m)/s)**4)
    k = total/n - 3
    assert k >= -2
    return k


# === Other statistical formulae ===

def sterrmean(s, n, N=None):
    """sterrmean(s, n [, N]) -> standard error of the mean.

    Return the standard error of the mean, optionally with a correction for
    finite population. Arguments given are:

    s: the standard deviation of the sample
    n: the size of the sample
    N (optional): the size of the population, or None

    If the sample size n is larger than (approximately) 5% of the population,
    it is necessary to make a finite population correction. To do so, give
    the argument N, which must be larger than or equal to n.

    >>> sterrmean(2, 16)
    0.5
    >>> sterrmean(2, 16, 21)
    0.25

    """
    stats.utils._validate_int(n)
    if n < 0:
        raise stats.StatsError('cannot have negative sample size')
    if N is not None:
        stats.utils._validate_int(N)
        if N < n:
            raise stats.StatsError('population size must be at least sample size')
    if s < 0.0:
        raise stats.StatsError('cannot have negative standard deviation')
    if n == 0:
        if N == 0: return float('nan')
        else: return float('inf')
    sem = s/math.sqrt(n)
    if N is not None:
        # Finite population correction.
        f = (N - n)/(N - 1)  # FPC squared.
        assert 0 <= f <= 1
        sem *= math.sqrt(f)
    return sem


# Tabachnick and Fidell (1996) appear to be the most commonly quoted
# source for standard error of skewness and kurtosis; see also "Numerical
# Recipes in Pascal", by William H. Press et al (Cambridge University Press).

def sterrskewness(n):
    """sterrskewness(n) -> float

    Return the approximate standard error of skewness for a sample of size
    n taken from an approximately normal distribution.

    >>> sterrskewness(15)  #doctest: +ELLIPSIS
    0.63245553203...

    """
    stats.utils._validate_int(n)
    if n == 0:
        return float('inf')
    return math.sqrt(6/n)


def sterrkurtosis(n):
    """sterrkurtosis(n) -> float

    Return the approximate standard error of kurtosis for a sample of size
    n taken from an approximately normal distribution.

    >>> sterrkurtosis(15)  #doctest: +ELLIPSIS
    1.2649110640...

    """
    stats.utils._validate_int(n)
    if n == 0:
        return float('inf')
    return math.sqrt(24/n)


# === Statistics of circular quantities ===

def circular_mean(data, deg=True):
    """Return the mean of circular quantities such as angles.

    Taking the mean of angles requires some care. Consider the mean of 15
    degrees and 355 degrees. The conventional mean of the two would be 185
    degrees, but a better result would be 5 degrees. This matches the result
    of averaging 15 and -5 degrees, -5 being equivalent to 355.

    >>> circular_mean([15, 355])  #doctest: +ELLIPSIS
    4.9999999999...

    If optional argument deg is a true value (the default), the angles are
    interpreted as degrees, otherwise they are interpreted as radians:

    >>> pi = math.pi
    >>> circular_mean([pi/4, -pi/4], False)
    0.0
    >>> # Exact value of the following is pi/12
    ... circular_mean([pi/3, 2*pi-pi/6], False)  #doctest: +ELLIPSIS
    0.261799387799...

    """
    ap = stats.add_partial
    if deg:
        data = (math.radians(theta) for theta in data)
    n, cosines, sines = 0, [], []
    for n, theta in enumerate(data, 1):
        ap(math.cos(theta), cosines)
        ap(math.sin(theta), sines)
    if n == 0:
        raise stats.StatsError(
        'circular mean of empty sequence is not defined')
    x = math.fsum(cosines)/n
    y = math.fsum(sines)/n
    theta = math.atan2(y, x)  # Note the order is swapped.
    if deg:
        theta = math.degrees(theta)
    return theta

