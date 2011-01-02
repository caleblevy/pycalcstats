#!/usr/bin/env python3

##  Copyright (c) 2011 Steven D'Aprano.
##  See the file __init__.py for the licence terms for this software.

"""
Univariate statistics.

"""

__all__ = [
    # Means and averages:
    'harmonic_mean', 'geometric_mean', 'quadratic_mean',
    # Other measures of central tendancy:
    'mode', 'moving_average',
    # Measures of spread:
    'average_deviation', 'median_average_deviation',
    # Other moments:
    'pearson_mode_skewness', 'skewness', 'kurtosis',
    # Sums and products:
    'product',
    # Assorted others:
    'sterrmean', 'stderrskewness', 'stderrkurtosis',
    # Statistics of circular quantities:
    'circular_mean',
    ]


import math
import operator
import functools
import itertools
import collections

import stats
from stats import StatsError


# Measures of central tendency (means and averages)
# -------------------------------------------------

def harmonic_mean(data):
    """Return the sample harmonic mean of a sequence of non-zero numbers.

    >>> harmonic_mean([0.25, 0.5, 1.0, 1.0])
    0.5

    The harmonic mean, or subcontrary mean, is the reciprocal of the
    arithmetic mean of the reciprocals of the data. It is best suited for
    averaging rates.
    """
    try:
        m = stats.mean(1.0/x for x in data)
    except ZeroDivisionError:
        # FIXME need to preserve the sign of the zero?
        # FIXME is it safe to assume that if data contains 1 or more zeroes,
        # the harmonic mean must itself be zero?
        return 0.0
    if m == 0.0:
        return math.copysign(float('inf'), m)
    return 1/m


def geometric_mean(data):
    """Return the sample geometric mean of a sequence of positive numbers.

    >>> geometric_mean([1.0, 2.0, 6.125, 12.25])
    3.5

    The geometric mean is the Nth root of the product of the data. It is
    best suited for averaging exponential growth rates.
    """
    ap = add_partial
    log = math.log
    partials = []
    count = 0
    try:
        for x in data:
            count += 1
            ap(log(x), partials)
    except ValueError:
        if x < 0:
            raise StatsError('geometric mean of negative number')
        return 0.0
    if count == 0:
        raise StatsError('geometric mean of empty sequence is not defined')
    p = math.exp(math.fsum(partials))
    return pow(p, 1.0/count)


def quadratic_mean(data):
    """Return the sample quadratic mean of a sequence of numbers.

    >>> quadratic_mean([2, 2, 4, 5])
    3.5

    The quadratic mean, or root-mean-square (RMS), is the square root of the
    arithmetic mean of the squares of the data. It is best used when
    quantities vary from positive to negative.
    """
    return math.sqrt(stats.mean(x*x for x in data))


def mode(data):
    """Returns the single most common element of a sequence of numbers.

    >>> mode([5.0, 7.0, 2.0, 3.0, 2.0, 2.0, 1.0, 3.0])
    2.0

    Raises StatsError if there is no mode, or if it is not unique.

    The mode is commonly used as an average.
    """
    L = sorted(
        [(count, value) for (value, count) in count_elems(data).items()],
        reverse=True)
    if len(L) == 0:
        raise StatsError('no mode is defined for empty iterables')
    # Test if there are more than one modes.
    if len(L) > 1 and L[0][0] == L[1][0]:
        raise StatsError('no distinct mode')
    return L[0][1]


def moving_average(data, window=3):
    """Iterate over data, yielding the simple moving average with a fixed
    window size (defaulting to three).

    >>> list(moving_average([40, 30, 50, 46, 39, 44]))
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


# Measures of spread (dispersion or variability)
# ----------------------------------------------

def average_deviation(data, m=None):
    """average_deviation(data [, m]) -> average absolute deviation of data.

    Returns the average deviation of the sample data from the population
    centre m (usually the mean, or the median). If you know the population
    mean or median, pass it as the second element:

    >>> data = [2.0, 2.25, 2.5, 2.5, 3.25]  # A sample from a population
    >>> mu = 2.75                           # with a known mean.
    >>> average_deviation(data, mu)
    0.45

    If you don't know the centre location, you can estimate it by passing
    the sample mean or median instead. If m is not None, or not given, the
    sample mean is calculated from the data and used:

    >>> average_deviation(data)
    0.3

    """
    if m is None:
        data = as_sequence(data)
        m = mean(data)
    n, total = _generalised_sum(data, lambda x: abs(x-m))
    if n < 1:
        raise StatsError('average deviation requires at least 1 data point')
    return total/n


def median_average_deviation(data, m=None, sign=0, scale=1):
    """Return the median absolute deviation (MAD) of data.

    The MAD is the median of the absolute deviations from the median, and
    is approximately equivalent to half the IQR.

    >>> median_average_deviation([1, 1, 2, 2, 4, 6, 9])
    1

    Arguments are:

    data    Iterable of data values.
    m       Optional centre location, nominally the median. If m is not
            given, or is None, the median is calculated from data.
    sign    If sign = 0 (the default), the ordinary median is used, otherwise
            either the low-median or high-median are used. See the median()
            function for further details.
    scale   Optional scale factor, by default no scale factor is applied.

    The MAD can be used as a robust estimate for the standard deviation by
    multipying it by a scale factor. The scale factor can be passed directly
    as a numeric value, which is assumed to be positive but no check is
    applied. Other values accepted are:

    'normal'    Apply a scale factor of 1.4826, applicable to data from a
                normally distributed population.
    'uniform'   Apply a scale factor of approximately 1.1547, applicable
                to data from a uniform distribution.
    None, 'none' or not supplied:
                No scale factor is applied (the default).

    The MAD is a more robust measurement of spread than either the IQR or
    standard deviation, and is less affected by outliers. The MAD is also
    defined for distributions such as the Cauchy distribution which don't
    have a mean or standard deviation.
    """
    # Check for an appropriate scale factor.
    if isinstance(scale, str):
        f = median_average_deviation.scaling.get(scale.lower())
        if f is None:
            raise StatsError('unrecognised scale factor `%s`' % scale)
        scale = f
    elif scale is None:
        scale = 1
    if m is None:
        data = as_sequence(data)
        m = median(data, sign)
    med = median((abs(x - m) for x in data), sign)
    return scale*med

median_average_deviation.scaling = {
    # R defaults to the normal scale factor:
    # http://stat.ethz.ch/R-manual/R-devel/library/stats/html/mad.html
    'normal': 1.4826,
    # Wikpedia has a derivation of that constant:
    # http://en.wikipedia.org/wiki/Median_absolute_deviation
    'uniform': math.sqrt(4/3),
    'none': 1,
    }


# Other moments of the data
# -------------------------

def pearson_mode_skewness(mean, mode, stdev):
    """Return the Pearson Mode Skewness from the mean, mode and standard
    deviation of a data set.

    >>> pearson_mode_skewness(2.5, 2.25, 2.5)
    0.1

    """
    if stdev > 0:
        return (mean-mode)/stdev
    elif stdev == 0:
        return float('nan') if mode == mean else float('inf')
    else:
        raise StatsError("standard deviation cannot be negative")


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
        data = as_sequence(data)
        if m is None: m = mean(data)
        if s is None: s = stdev(data, m)
    n, total = _generalised_sum(data, lambda x: ((x-m)/s)**3)
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
    kurtosis has heavier tails and a sharper peak than normal; negative
    kurtosis has ligher tails and a flatter peak.

    There is no upper limit for kurtosis, and a lower limit of -2. Higher
    kurtosis means more of the variance is the result of infrequent extreme
    deviations, as opposed to frequent modestly sized deviations.

        :: CAUTION ::
        As a rule of thumb, a non-zero value for kurtosis should only
        be treated as meaningful if its absolute value is larger than
        approximately twice its standard error. See stderrkurtosis.

    """
    if m is None or s is None:
        data = as_sequence(data)
        if m is None: m = mean(data)
        if s is None: s = stdev(data, m)
    n, total = _generalised_sum(data, lambda x: ((x-m)/s)**4)
    k = total/n - 3
    assert k >= -2
    return k



# === Sums and products ===




def product(data, start=1):
    """Return the product of a sequence of numbers.

    >>> product([1, 2, -3, 2, -1])
    12

    If optional argument start is given, it is multiplied to the sequence.
    If the sequence is empty, start (defaults to 1) is returned.
    """
    # FIXME this doesn't seem to be numerically stable enough.
    return functools.reduce(operator.mul, data, start)
        # Note: do *not* be tempted to do something clever with logarithms:
        # return math.exp(sum([math.log(x) for x in data], start))
        # This is FAR less accurate than the naive multiplication above.


# === Partitioning, sorting and binning ===

def count_elems(data):
    """Count the elements of data, returning a Counter.

    >>> d = count_elems([1.5, 2.5, 1.5, 0.5])
    >>> sorted(d.items())
    [(0.5, 1), (1.5, 2), (2.5, 1)]

    """
    D = {}
    for element in data:
        D[element] = D.get(element, 0) + 1
    return D  #collections.Counter(data)


# === Trimming of data ===

"this section intentionally left blank"

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
    if N is not None and N < n:
        raise StatsError('population size must be at least sample size')
    if n < 0:
        raise StatsError('cannot have negative sample size')
    if s < 0.0:
        raise StatsError('cannot have negative standard deviation')
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
# Presumably "Numerical Recipes in C" and "... Fortran" by the same authors
# say the same thing.

def stderrskewness(n):
    """stderrskewness(n) -> float

    Return the approximate standard error of skewness for a sample of size
    n taken from an approximately normal distribution.

    >>> stderrskewness(15)  #doctest: +ELLIPSIS
    0.63245553203...

    """
    if n == 0:
        return float('inf')
    return math.sqrt(6/n)


def stderrkurtosis(n):
    """stderrkurtosis(n) -> float

    Return the approximate standard error of kurtosis for a sample of size
    n taken from an approximately normal distribution.

    >>> stderrkurtosis(15)  #doctest: +ELLIPSIS
    1.2649110640...

    """
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
    >>> theta = circular_mean([pi/3, 2*pi-pi/6], False)
    >>> theta  # Exact value is pi/12  #doctest: +ELLIPSIS
    0.261799387799...

    """
    ap = add_partial
    if deg:
        data = (math.radians(theta) for theta in data)
    n, cosines, sines = 0, [], []
    for n, theta in enumerate(data, 1):
        ap(math.cos(theta), cosines)
        ap(math.sin(theta), sines)
    if n == 0:
        raise StatsError('circular mean of empty sequence is not defined')
    x = math.fsum(cosines)/n
    y = math.fsum(sines)/n
    theta = math.atan2(y, x)  # Note the order is swapped.
    if deg:
        theta = math.degrees(theta)
    return theta

