#!/usr/bin/env python3
# -*- coding: utf8 -*-

##  Copyright (c) 2011 Steven D'Aprano.
##  See the file __init__.py for the licence terms for this software.

"""
Univariate statistics.

This module provides functions for calculating summary statistics for
univariate (single-variable) data. For additional functions, see the main
``stats`` module.


This module provides the following univariate statistics functions:

    Function            Description
    ==================  ===============================================
    average_deviation   Average deviation from a central location.
    circular_mean       Mean (average) of circular quantities.
    geometric_mean*     Mean of exponential growth rates.
    harmonic_mean*      Mean of rates or speeds.
    kurtosis*           Measure of shape of the data.
    mode                Most frequent value.
    moving_average      Simple moving average iterator.
    pearson_skewness    Measure of symmetry of the data.
    pkurtosis*          Population kurtosis.
    pskewness*          Population skewness.
    quadratic_mean*     Root-mean-square average.
    skewness*           Measure of the symmetry of the data
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

import decimal
import math
import operator
import functools
import itertools
import collections

import stats
import stats.utils
import stats.vectorize as v


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
    try:
        return num/den
    except (decimal.DivisionByZero, decimal.InvalidOperation):
        # num and den could be NANs, INFs, or den == 0. The easiest way
        # to handle all the cases is just do the division again.
        with decimal.localcontext() as ctx:
            ctx.traps[decimal.DivisionByZero] = 0
            ctx.traps[decimal.InvalidOperation] = 0
            return num/den
    except ZeroDivisionError:
        assert den == 0  # Division by NAN or INF is handled okay.
        assert not math.isnan(num)  # NAN/x will not raise Zero
        if num == 0:
            return float('nan')
        else:
            result = math.copysign(float('inf'), den)  # Support signed zero.
            if num < 0:
                result = -result
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
    n, total = stats._len_sum(v.apply(invert, x) for x in data)
    if not n:
        raise stats.StatsError(
        'harmonic mean of empty sequence is not defined')
    return v.div(n, total)


def geometric_mean(data):
    """Return the sample geometric mean of a sequence of non-negative numbers.

    >>> geometric_mean([1.0, 2.0, 6.125, 12.25])
    3.5

    The geometric mean of N items is the Nth root of the product of the
    items. It is best suited for averaging exponential growth rates.

    If data is an iterable of sequences, each inner sequence represents a
    row of data, and the geometric mean of each column is returned. Every
    row must have the same number of columns, or ValueError is raised.

    >>> data = [[1, 1],
    ...         [2, 3],
    ...         [3, 9]]
    ...
    >>> geometric_mean(data)  #doctest: +ELLIPSIS
    [1.81712059283..., 3.0]

    """
    # Calculate the length and product of data.
    def safe_mul(a, b):
        x = a*b
        if x < 0: return float('nan')
        return x
    mul = functools.partial(v.apply_op, safe_mul)

    # Special case for speed.
    if isinstance(data, list):
        n = len(data)
    else:
        n = None
        data = stats._countiter(data)
    prod = functools.reduce(mul, data)
    if n is None:
        n = data.count
    if not n:
        raise stats.StatsError(
        'geometric mean of empty sequence is not defined')
    return v.pow(prod, 1.0/n)


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
    count, total = stats._len_sum(v.sqr(x) for x in data)
    if not count:
        raise stats.StatsError(
        'quadratic mean of empty sequence is not defined')
    return v.sqrt(v.div(total, count))


def mode(data):
    """Returns the most common element of a sequence of discrete numbers.

    The mode is commonly used as an average. It is the "most typical"
    value of a distribution or data set. This function assumes that the
    values are discrete and exact, and returns the most frequent value:

    >>> mode([5, 7, 2, 3, 2, 2, 1, 3])
    2

    If there is no such element, or it is not unique, ``StatsError`` is
    raised.

    ``mode`` also works on nominal data:

    >>> mode(['big', 'small', 'medium', 'small', 'huge', 'small', 'medium'])
    'small'

    If your data is continuous, see functions .... FIXME
    """
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

    If data is an iterable of sequences, each inner sequence represents a
    row of data, and ``average_deviation`` operates on each column. Every
    row must have the same number of columns, or ValueError is raised.
    Similarly, m (if given) must have either the same number of items, or
    be a single number.

    >>> data = [[0, 1, 2, 4],
    ...         [1, 2, 4, 6],
    ...         [2, 4, 6, 6]]
    ...
    >>> average_deviation(data, [1, 2, 3.5, 6])  #doctest: +ELLIPSIS
    [0.666666..., 1.0, 1.5, 0.666666...]

    """
    if m is None:
        if not isinstance(data, list):
            data = list(data)
        m = stats.mean(data)
    f = lambda x, m: abs(x-m)
    count, total = stats._len_sum(v.apply(f, x, m) for x in data)
    if not count:
        raise stats.StatsError(
        'average deviation requires at least 1 data point')
    return v.div(total, count)


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


def pskewness(data, m=None, s=None):
    """pskewness(data [,m [,s]]) -> population skewness of data.

    This returns γ₁ "\\N{GREEK SMALL LETTER GAMMA}\\N{SUBSCRIPT ONE}", the
    population skewness. For more information about skewness, see the sample
    skewness function ``skewness``.

    >>> pskewness([1.25, 1.5, 1.5, 1.75, 1.75, 2.5, 2.75, 4.5])
    ... #doctest: +ELLIPSIS
    1.37474650254...

    """
    n, total = stats._std_moment(data, m, s, 3)
    assert n >= 0
    if n <= 1:
        raise StatsError('no skewness is defined for empty data')
    return v.div(total, n)


def skewness(data, m=None, s=None):
    """skewness(data [,m [,s]]) -> sample skewness of data.

    The skewness, or third standardised moment, of data is the degree to
    which it is skewed to the left or right of the mean.

    This returns g₁ "g\\N{SUBSCRIPT ONE}", the sample skewness. For the
    population skewness, see function ``pskewness``.

        WARNING: The mathematical terminology and notation related to
        skewness is often inconsistent and contradictory. See Wolfram
        Mathworld for further details:

        http://mathworld.wolfram.com/Skewness.html

    >>> skewness([1.25, 1.5, 1.5, 1.75, 1.75, 2.5, 2.75, 4.5])
    ... #doctest: +ELLIPSIS
    1.71461013539878...

    If you already know one or both of the population mean and standard
    deviation, you can pass the mean as optional argument m and/or the
    standard deviation as s:

    >>> skewness([1.25, 1.5, 1.5, 1.75, 1.75, 2.5, 2.75, 4.5], m=2.25, s=1)
    ... #doctest: +ELLIPSIS
    1.47132881615329...

        CAUTION: "Garbage in, garbage out" applies here. You can pass
        any values you like as ``m`` or ``s``, but if they are not
        sensible estimates for the mean and standard deviation, the
        result returned as the skewness will likewise not be sensible.

    If m or s are not given, or are None, they are estimated from the data.

    If data is an iterable of sequences, each inner sequence represents a
    row of data, and ``skewness`` operates on each column. Every row must
    have the same number of columns, or ValueError is raised.

    >>> data = [[0, 1],
    ...         [1, 5],
    ...         [2, 6],
    ...         [5, 7]]
    ...
    >>> skewness(data)  #doctest: +ELLIPSIS
    [1.19034012827899..., -1.44305883553164...]

    Similarly, if either m or s are given, they must be either a single
    number or have the same number of items as the data:

    >>> skewness(data, m=[2.5, 5.0], s=2)  #doctest: +ELLIPSIS
    [-0.189443057077845..., -2.97696232550900...]

    A negative skewness indicates that the distribution's left-hand tail is
    longer than the tail on the right-hand side, and that the majority of
    the values (including the median) are to the right of the mean. A
    positive skew indicates that the right-hand tail is longer, and that the
    majority of values are to the left of the mean. A zero skew indicates
    that the values are evenly distributed around the mean, often but not
    necessarily implying the distribution is symmetric.

        CAUTION: As a rule of thumb, a non-zero value for skewness
        should only be treated as meaningful if its absolute value is
        larger than approximately twice its standard error. See also
        ``stderrskewness``.

    """
    n, total = stats._std_moment(data, m, s, 3)
    assert n >= 0
    if n < 3:
        raise StatsError('sample skewness requires at least three items')
    skew = v.div(total, n)
    k = math.sqrt(n*(n-1))/(n-2)
    return v.mul(k, skew)


def pkurtosis(data, m=None, s=None):
    """pkurtosis(data [,m [,s]]) -> population kurtosis of data.

    This returns γ₂ "\\N{GREEK SMALL LETTER GAMMA}\\N{SUBSCRIPT TWO}", the
    population kurtosis relative to that of the normal distribution, also
    known as the excess kurtosis. For the "kurtosis proper" known as
    β₂ "\\N{GREEK SMALL LETTER BETA}\\N{SUBSCRIPT TWO}", add 3 to the result.

    For more information about kurtosis, see the sample kurtosis function
    ``kurtosis``.

    >>> pkurtosis([1.25, 1.5, 1.5, 1.75, 1.75, 2.5, 2.75, 4.5])
    ... #doctest: +ELLIPSIS
    0.7794232987...

    """
    n, total = stats._std_moment(data, m, s, 4)
    assert n >= 0
    assert total >= 1
    if n <= 1:
        raise StatsError('no kurtosis is defined for empty data')
    kurt = v.div(total, n)
    return v.sub(kurt, 3)


def kurtosis(data, m=None, s=None):
    """kurtosis(data [,m [,s]]) -> sample excess kurtosis of data.

    The kurtosis of a distribution is a measure of its shape. This function
    returns an estimate of the sample excess kurtosis usually known as g₂
    "g\\N{SUBSCRIPT TWO}". For the population kurtosis, see ``pkurtosis``.

        WARNING: The mathematical terminology and notation related to
        kurtosis is often inconsistent and contradictory. See Wolfram
        Mathworld for further details:

        http://mathworld.wolfram.com/Kurtosis.html

    >>> kurtosis([1.25, 1.5, 1.5, 1.75, 1.75, 2.5, 2.75, 4.5])
    ... #doctest: +ELLIPSIS
    3.03678892733564...

    If you already know one or both of the population mean and standard
    deviation, you can pass the mean as optional argument m and/or the
    standard deviation as s:

    >>> kurtosis([1.25, 1.5, 1.5, 1.75, 1.75, 2.5, 2.75, 4.5], m=2.25, s=1)
    2.3064453125

        CAUTION: "Garbage in, garbage out" applies here. You can pass
        any values you like as ``m`` or ``s``, but if they are not
        sensible estimates for the mean and standard deviation, the
        result returned as the kurtosis will likewise not be sensible.
        If you give either m or s, and the calculated kurtosis is out
        of range, a warning is raised.

    If m or s are not given, or are None, they are estimated from the data.

    If data is an iterable of sequences, each inner sequence represents a
    row of data, and ``kurtosis`` operates on each column. Every row must
    have the same number of columns, or ValueError is raised.

    >>> data = [[0, 1],
    ...         [1, 5],
    ...         [2, 6],
    ...         [5, 7]]
    ...
    >>> kurtosis(data)  #doctest: +ELLIPSIS
    [1.50000000000000..., 2.23486717956161...]

    Similarly, if either m or s are given, they must be either a single
    number or have the same number of items:

    >>> kurtosis(data, m=[3, 5], s=2)  #doctest: +ELLIPSIS
    [-0.140625, 18.4921875]

    The kurtosis of a population is a measure of the peakedness and weight
    of the tails. The normal distribution has kurtosis of zero; positive
    kurtosis generally has heavier tails and a sharper peak than normal;
    negative kurtosis generally has lighter tails and a flatter peak.

    There is no upper limit for kurtosis, and a lower limit of -2. Higher
    kurtosis means more of the variance is the result of infrequent extreme
    deviations, as opposed to frequent modestly sized deviations.

        CAUTION: As a rule of thumb, a non-zero value for kurtosis
        should only be treated as meaningful if its absolute value is
        larger than approximately twice its standard error. See also
        ``stderrkurtosis``.

    """
    n, total = stats._std_moment(data, m, s, 4)
    assert n >= 0
    v.assert_(lambda x: x >= 1, total)
    if n < 4:
        raise StatsError('sample kurtosis requires at least 4 data points')
    q = (n-1)/((n-2)*(n-3))
    gamma2 = v.div(total, n)
    # Don't do this:-
    # kurt = v.mul((n+1)*q, gamma2)
    # kurt = v.sub(kurt, 3*(n-1)*q)
    #   Even though the above two commented out lines are mathematically
    #   equivalent to the next two, and cheaper, they appear to be
    #   slightly less accurate.
    kurt = v.sub(v.mul(n+1, gamma2), 3*(n-1))
    kurt = v.mul(q, kurt)
    if v.isiterable(kurt): out_of_range = any(x < -2 for x in kurt)
    else: out_of_range = kurt < -2
    if m is s is None:
        assert not out_of_range, 'kurtosis failed: %r' % kurt
        # This is a "should never happen" condition, hence an assertion.
    else:
        # This, on the other hand, can easily happen if the caller
        # gives junk values for m or s. The difference between a junk
        # value and a legitimate value can be surprisingly subtle!
        if out_of_range:
            import warnings
            warnings.warn('calculated kurtosis out of range')
    return kurt


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
# Mathworld also references Kendall et al. (1998).

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

