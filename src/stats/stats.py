#!/usr/bin/env python3



# Implementation note re doctests:
# Doctest examples have been either calculated by hand (for simple examples)
# or calculated with a HP-48GX calculator (for less simple examples.)


__all__ = [
    # Means and averages:
    'mean', 'harmonic_mean', 'geometric_mean', 'quadratic_mean', 'median',
    'mode', 'midrange',
    # Measures of spread:
    'stdev', 'pstdev', 'variance', 'pvariance', 'range', 'iqr',
    'average_deviation',
    # Multivariate statistics:
    'corr', 'cov', 'pcov', 'errsumsq', 'linr',
    # Sums and products:
    'sum', 'sumsq', 'product', 'xsums', 'xysums', 'Sxx', 'Syy', 'Sxy',
    # Assorted others:
    'sterrmean', 'StatsError', 'minmax',
    ]

import math
import operator
import functools
import itertools
import collections


# === Exceptions ===

class StatsError(ValueError):
    pass


# === Utility functions and classes ===


def sorted_data(func):
    """Decorator to sort data passed to stats functions."""
    @functools.wraps(func)
    def inner(data):
        if isinstance(data, list):
            # Sort in place.
            data.sort()
        else:
            data = sorted(data)
        return func(data)
    return inner


def minmax(*values, **kw):
    """minmax(iterable [, key=func]) -> (minimum, maximum)
    minmax(a, b, c, ... [key=func]) -> (minimum, maximum)

    With a single iterable argument, return a two-tuple of its smallest
    item and largest item. With two or more arguments, return the smallest
    and largest arguments.
    """
    if len(values) == 1:
        values = values[0]
    if isinstance(values, collections.Sequence):
        # For speed, fall back on built-in min and max functions when
        # data is a sequence and can be safely iterated over twice.
        minimum = min(values, **kw)
        maximum = max(values, **kw)
    else:
        # Iterator argument, so fall back on a slow pure-Python solution.
        raise NotImplementedError('not yet implemented')
    return (minimum, maximum)


# Modified from http://code.activestate.com/recipes/393090/
def add_partial(x, partials):
    """Helper function for full-precision summation of binary floats.

    Adds x in place to the list partials.
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


def makeseq(data):
    """Helper function to convert iterable arguments into sequences."""
    if isinstance(data, (list, tuple)): return data
    return list(data)


# === Basic univariate statistics ===


# Measures of central tendency (means and averages)
# -------------------------------------------------


def mean(data):
    """Return the sample arithmetic mean of a sequence of numbers.

    >>> mean([1.0, 2.0, 3.0, 4.0])
    2.5

    The arithmetic mean is the sum of the data divided by the number of data.
    It is commonly called "the average".
    """
    # Fast path for sequence data.
    try:
        n = len(data)
    except TypeError:
        # Slower path for iterable data with no len.
        ap = add_partial
        partials = []
        n = 0
        for x in data:
            ap(x, partials)
            n += 1
        sumx = math.fsum(partials)
    else:
        sumx = sum(data)  # Not the built-in version.
    if n == 0:
        raise StatsError('no data')
    return sumx/n


def harmonic_mean(data):
    """Return the sample harmonic mean of a sequence of non-zero numbers.

    >>> harmonic_mean([0.25, 0.5, 1.0, 1.0])
    0.5

    The harmonic mean, or subcontrary mean, is the reciprocal of the
    arithmetic mean of the reciprocals of the data. It is best suited for
    averaging rates.
    """
    try:
        m = mean(1.0/x for x in data)
    except ZeroDivisionError:
        # FIXME need to preserve the sign of the zero?
        # FIXME is it safe to assume that if data contains 1 or more zeroes,
        # the harmonic mean must itself be zero?
        return 0.0
    # FIXME if m is zero, the following will raise ZeroDivisionError. Would
    # it be better to return float('inf') (plus or minus)?
    return 1.0/m


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
        raise StatsError('no data')
    p = math.exp(math.fsum(partials))
    return pow(p, 1.0/count)


def quadratic_mean(data):
    """Return the sample quadratic mean of a sequence of numbers.

    >>> quadratic_mean([2, 2, 4, 5])
    3.5

    The quadratic mean, or root-mean-square (RMS) is the square root of the
    arithmetic mean of the squares of the data. It is best used when
    quantities vary from positive to negative.
    """
    return math.sqrt(mean(x*x for x in data))


@sorted_data
def median(data):
    """Returns the median (middle) value of a sequence of numbers.

    >>> median([3.0, 5.0, 2.0])
    3.0

    The median is the middle data point in a sorted sequence of values. If
    the argument to median is a list, it will be sorted in place, otherwise
    the values will be collected into a sorted list.

    The median is commonly used as an average. It is more robust than the
    mean for data that contains outliers. The median is equivalent to the
    second quartile or the 50th percentile.
    """
    n = len(data)
    n2 = n//2
    if n%2 == 1:
        # For an odd number of items, we take the middle one.
        return data[n2]
    else:
        # If there are an even number of items, we take the average
        # of the two middle ones.
        return (data[n2-1] + data[n2])/2


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
        raise StatsError('no data')
    # Test if there are more than one modes.
    if len(L) > 1 and L[0][0] == L[1][0]:
        raise StatsError('no distinct mode')
    return L[0][1]


def midrange(data):
    """Returns the midrange of a sequence of numbers.

    >>> midrange([2.0, 4.5, 7.5])
    4.75

    The midrange is halfway between the smallest and largest element. It is
    a weak measure of central tendency.
    """
    a, b = minmax(data)
    return (a + b)/2


# Quartiles, deciles and percentiles
# ----------------------------------


@sorted_data
def quartiles(data):
    """Return (Q1, Q2, Q3) for data.

    Returns a tuple of the first quartile Q1, the second quartile Q2 (also
    known as the median) and the third quartile Q3 from sortable sequence
    data.
    """
    raise NotImplementedError('not implemented yet')


# Measures of spread (dispersion or variability)
# ----------------------------------------------


def stdev(data):
    """Return the sample standard deviation of data.

    >>> stdev([1.5, 2.5, 2.5, 2.75, 3.25, 4.75])  #doctest: +ELLIPSIS
    1.08108741552...

    If data represents the entire population, and not just a sample, then
    use pstdev instead.
    """
    return math.sqrt(variance(data))


def pstdev(data):
    """Return the population standard deviation of data.

    >>> pstdev([1.5, 2.5, 2.5, 2.75, 3.25, 4.75])  #doctest: +ELLIPSIS
    0.986893273527...

    You should use pstdev when data represents the entire population rather
    than a statistical sampling.
    """
    return math.sqrt(pvariance(data))


def variance(data):
    """Return the sample variance of data.

    >>> variance([0.25, 0.5, 1.25, 1.25,
    ...           1.75, 2.75, 3.5])  #doctest: +ELLIPSIS
    1.37202380952...

    If data represents the entire population, and not just a sample, then
    use pvariance instead.
    """
    t = xsums(data)
    return t.Sxx/(t.n*(t.n-1))


def pvariance(data):
    """Return the population variance of data.

    >>> pvariance([0.25, 0.5, 1.25, 1.25,
    ...            1.75, 2.75, 3.5])  #doctest: +ELLIPSIS
    1.17602040816...

    You should use pvariance when data represents the entire population rather
    than a statistical sampling.
    """
    t = xsums(data)
    return t.Sxx/(t.n**2)


def range(data):
    """Return the range of a sequence of numbers.

    >>> range([1.0, 3.5, 7.5, 2.0, 0.25])
    7.25

    The range is the difference between the smallest and largest element. It
    is a weak measure of statistical variability.
    """
    a, b = minmax(data)
    return b - a


def iqr(data):
    """Returns the Inter-Quartile Range of a sequence of numbers.

    The IQR is the difference between the first and third quartile.
    """
    q1, q2, q3 = quartiles(data)
    return q3 - q1


def average_deviation(xdata, Mx=None):
    """Return the average absolute deviation of data.

    xdata = iterable of data values
    Mx = measure of central tendency for xdata.

    Mx is usually chosen to be the mean or median. If Mx is not given, or
    is None, the mean is calculated from xdata and that value is used.
    """
    if Mx is None:
        xdata = makeseq(xdata)
        Mx = mean(xdata)
    ap = add_partial
    partials = []
    n = 0
    for x in xdata:
        n += 1
        ap(abs(x - Mx), partials)
    return math.fsum(partials)/n


# === Simple multivariate statistics ===


def corr(xdata, ydata=None):
    """Return the sample Pearson's Correlation Coefficient of (x,y) data.

    >>> corr([(0.1, 2.3), (0.5, 2.7), (1.2, 3.1),
    ...       (1.7, 2.9)])  #doctest: +ELLIPSIS
    0.827429009335...

    The Pearson correlation is +1 in the case of a perfect positive
    correlation (i.e. an increasing linear relationship), -1 in the case of
    a perfect anti-correlation (i.e. a decreasing linear relationship), and
    some value between -1 and 1 in all other cases, indicating the degree
    of linear dependence between the variables.

    >>> xdata = [0.0, 0.1, 0.25, 1.2, 1.75]
    >>> ydata = [2.5*x + 0.3 for x in xdata]  # Perfect correlation.
    >>> corr(xdata, ydata)
    1.0
    >>> corr(xdata, [10-y for y in ydata])  # Perfect anti-correlation.
    -1.0

    """
    t = xysums(xdata, ydata)
    r = t.Sxy/math.sqrt(t.Sxx*t.Syy)
    # FIXME sometimes r is just slightly out of range. (Rounding error?)
    # In the absence of any better idea of how to fix it, hit it on the head.
    if r > 1.0:
        assert (r - 1.0) <= 1e-15, 'r out of range (> 1.0)'
        r = 1.0
    elif r < -1.0:
        assert (r + 1.0) >= -1e-15, 'r out of range (< -1.0)'
        r = -1.0
    return r


def cov(xdata, ydata=None):
    """Return the sample covariance between (x, y) data.

    >>> cov([(0.1, 2.3), (0.5, 2.7), (1.2, 3.1),
    ...      (1.7, 2.9)])  #doctest: +ELLIPSIS
    0.201666666666...

    >>> print(cov([0.75, 1.5, 2.5, 2.75, 2.75], [0.25, 1.1, 2.8, 2.95, 3.25]))
    1.1675
    >>> cov([(0.1, 2.3), (0.5, 2.7), (1.2, 3.1), (1.7, 2.9)])  #doctest: +ELLIPSIS
    0.201666666666...

    Covariance reduces down to standard variance when applied to the same data
    as both the x and y values:

    >>> data = [1.2, 0.75, 1.5, 2.45, 1.75]
    >>> print(cov(data, data))
    0.40325
    >>> print(variance(data))
    0.40325

    """
    t = xysums(xdata, ydata)
    return t.Sxy/(t.n*(t.n-1))


def pcov(xdata, ydata=None):
    """Return the population covariance between (x, y) data.

    >>> print(pcov([0.75, 1.5, 2.5, 2.75, 2.75], [0.25, 1.1, 2.8, 2.95, 3.25]))
    0.934
    >>> print(pcov([(0.1, 2.3), (0.5, 2.7), (1.2, 3.1), (1.7, 2.9)]))
    0.15125

    """
    t = xysums(xdata, ydata)
    return t.Sxy/(t.n**2)


def errsumsq(xdata, ydata=None):
    """Return the error sum of squares of (x,y) data."""
    t = xysums(xdata, ydata)
    return (t.Sxx*t.Syy - (t.Sxy**2))/(t.n*(t.n-2)*t.Sxx)


def linr(xdata, ydata=None):
    """Return the linear regression coefficients a and b for (x,y) data.

    >>> xdata = [0.0, 0.25, 1.25, 1.75, 2.5, 2.75]
    >>> ydata = [1.5*x + 0.25 for x in xdata]
    >>> linr(xdata, ydata)  #doctest: +ELLIPSIS
    (0.25, 1.5)

    """
    t = xysums(xdata, ydata)
    b = t.Sxy/t.Sxx
    a = t.sumy/t.n - b*t.sumx/t.n
    return (a, b)


# === Sums and products ===


def sum(data):
    """Return a full-precision sum of a sequence of numbers.

    >>> sum([2.25, 4.5, -0.5, 1.0])
    7.25

    Due to round-off error, the builtin sum can suffer from catastrophic
    cancellation, e.g. sum([1, 1e100, 1, -1e100] * 10000) => 0.0
    This version avoids that error:

    >>> sum([1, 1e100, 1, -1e100] * 10000)
    20000.0

    """
    return math.fsum(data)


def product(data):
    """Return the product of a sequence of numbers.

    >>> product([1, 2, -3, 2, -1])
    12

    """
    return functools.reduce(operator.mul, data)


def sumsq(data):
    """Return the sum of the squares of a sequence of numbers.

    >>> sumsq([2.25, 4.5, -0.5, 1.0])
    26.5625

    """
    return sum(x*x for x in data)


def Sxx(xdata):
    """Return Sxx from (x,y) data or x data alone.

    Sxx(xdata) -> n*sum(x**2) - sum(x)**2

    xdata can be either a sequence of x values alone, or a sequence of (x,y)
    values. In the later case, the y values will be ignored.

    If you need all three of Sxx, Syy and Sxy, it is more efficient to use
    xysums() instead.
    """
    data = iter(xdata)
    first = next(data)
    if isinstance(first, tuple):
        if len(first) != 2:
            raise ValueError(
            'expected 2-tuple (x,y) but got %d items instead' % len(first))
        data = itertools.chain([first[0]], (x[0] for x in data))
    else:
        data = itertools.chain([first], data)
    return Sxy(*itertools.tee(data))


def Syy(ydata):
    """Return Syy from (x,y) data or y data alone.

    Syy(ydata) -> n*sum(y**2) - sum(y)**2

    ydata can be either a sequence of y values alone, or a sequence of (x,y)
    values. In the later case, the x values will be ignored.

    If you need all three of Sxx, Syy and Sxy, it is more efficient to use
    xysums() instead.
    """
    data = iter(ydata)
    first = next(data)
    if isinstance(first, tuple):
        if len(first) != 2:
            raise ValueError(
            'expected 2-tuple (x,y) but got %d items instead' % len(first))
        data = itertools.chain([first[1]], (x[1] for x in data))
    else:
        data = itertools.chain([first], data)
    return Sxy(*itertools.tee(data))


def Sxy(xdata, ydata=None):
    """Return Sxy from (x,y) data.

    Sxy(xdata, ydata) -> n*sum(x*y) - sum(x)*sum(y)

    If ydata is given, both it and xdata must be sequences of numeric values.
    They will be truncated to the shorter of the two. If ydata is not given,
    xdata must be a sequence of (x,y) pairs.

    If you need all three of Sxx, Syy and Sxy, it is more efficient to use
    xysums() instead.
    """
    if ydata is None:
        data = xdata
    else:
        data = zip(xdata, ydata)
    n = 0
    sumx, sumy, sumxy = [], [], []
    ap = add_partial
    for x, y in data:
        n += 1
        ap(x, sumx)
        ap(y, sumy)
        ap(x*y, sumxy)
    sumx = math.fsum(sumx)
    sumy = math.fsum(sumy)
    sumxy = math.fsum(sumxy)
    return n*sumxy - sumx*sumy


def xsums(xdata):
    """Return statistical sums from x data.

    xsums(xdata) -> tuple of sums with named fields

    Returns a named tuple with four fields:

        Name    Description
        ======  ==========================
        n:      number of data items
        sumx:   sum of x values
        sumx2:  sum of x-squared values
        Sxx:    n*(sumx2) - (sumx)**2

    Note that the last field is named with an initial uppercase S, to match
    the standard statistical term.

    >>> tuple(xsums([2.0, 1.5, 4.75]))
    (3, 8.25, 28.8125, 18.375)

    This function calculates all the sums with one pass over the data, and so
    is more efficient than calculating the individual fields one at a time.
    """
    ap = add_partial
    n = 0
    sumx, sumx2 = [], []
    for x in xdata:
        n += 1
        ap(x, sumx)
        ap(x*x, sumx2)
    sumx = math.fsum(sumx)
    sumx2 = math.fsum(sumx2)
    Sxx = n*sumx2 - sumx*sumx
    statsums = collections.namedtuple('statsums', 'n sumx sumx2 Sxx')
    return statsums(*(n, sumx, sumx2, Sxx))


def xysums(xdata, ydata=None):
    """Return statistical sums from x,y data pairs.

    xysums(xdata, ydata) -> tuple of sums with named fields
    xysums(xydata) -> tuple of sums with named fields

    Returns a named tuple with nine fields:

        Name    Description
        ======  ==========================
        n:      number of data items
        sumx:   sum of x values
        sumy:   sum of y values
        sumxy:  sum of x*y values
        sumx2:  sum of x-squared values
        sumy2:  sum of y-squared values
        Sxx:    n*(sumx2) - (sumx)**2
        Syy:    n*(sumy2) - (sumy)**2
        Sxy:    n*(sumxy) - (sumx)*(sumy)

    Note that the last three fields are named with an initial uppercase S,
    to match the standard statistical term.

    This function calculates all the sums with one pass over the data, and so
    is more efficient than calculating the individual fields one at a time.

    If ydata is missing or None, xdata must be an iterable of pairs of numbers
    (x,y). Alternately, both xdata and ydata can be iterables of numbers, which
    will be truncated to the shorter of the two.
    """
    if ydata is None:
        data = xdata
    else:
        data = zip(xdata, ydata)
    ap = add_partial
    n = 0
    sumx, sumy, sumxy, sumx2, sumy2 = [], [], [], [], []
    for x, y in data:
        n += 1
        ap(x, sumx)
        ap(y, sumy)
        ap(x*y, sumxy)
        ap(x*x, sumx2)
        ap(y*y, sumy2)
    sumx = math.fsum(sumx)
    sumy = math.fsum(sumy)
    sumxy = math.fsum(sumxy)
    sumx2 = math.fsum(sumx2)
    sumy2 = math.fsum(sumy2)
    Sxx = n*sumx2 - sumx*sumx
    Syy = n*sumy2 - sumy*sumy
    Sxy = n*sumxy - sumx*sumy
    statsums = collections.namedtuple(
        'statsums', 'n sumx sumy sumxy sumx2 sumy2 Sxx Syy Sxy')
    return statsums(*(n, sumx, sumy, sumxy, sumx2, sumy2, Sxx, Syy, Sxy))


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
    return collections.Counter(data)


# === Trimming of data ===

"this section intentionally left blank"

# === Other statistical formulae ===

def sterrmean(s, n):
    """sterrmean(s, n) -> float

    Return the standard error of the mean.

    >>> sterrmean(2, 16)
    0.5

    s = standard deviation of the sample
    n = number of items in the sample
    """
    return s/math.sqrt(n)





if __name__ == '__main__':
    import doctest
    doctest.testmod()


