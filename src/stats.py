#!/usr/bin/env python3

##  Package stats.py
##
##  Copyright (c) 2010 Steven D'Aprano.
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
'Scientific calculator' statistics for Python 3.

>>> mean([-1.0, 2.5, 3.25, 5.75])
2.625
>>> data = iter([-1.0, 2.5, 3.25, 5.75, 0.0, 3.75])
>>> mean(data)
2.375

"""


# Module metadata.
__version__ = "0.1.1a"
__date__ = "2010-11-14"
__author__ = "Steven D'Aprano"
__author_email__ = "steve+python@pearwood.info"



__all__ = [
    # Means and averages:
    'mean', 'harmonic_mean', 'geometric_mean', 'quadratic_mean',
    # Other measures of central tendancy:
    'median', 'mode', 'midrange', 'trimean',
    # Moving averages:
    'running_average', 'weighted_running_average', 'simple_moving_average',
    # Other point statistics:
    'quartiles', 'quantile', 'decile', 'percentile', 'boxwhiskerplot',
    # Measures of spread:
    'pvariance', 'variance', 'pstdev', 'stdev',
    'pvariance1', 'variance1', 'pstdev1', 'stdev1',
    'range', 'iqr', 'average_deviation', 'median_average_deviation',
    # Other moments:
    'quartile_skewness', 'pearson_mode_skewness', 'skewness', 'kurtosis',
    # Multivariate statistics:
    'qcorr', 'corr', 'corr1', 'pcov', 'cov', 'errsumsq', 'linr',
    # Sums and products:
    'sum', 'sumsq', 'product', 'Sxx', 'Syy', 'Sxy',
    # Assorted others:
    'StatsError', 'sterrmean', 'minmax',
    # Statistics of circular quantities:
    'circular_mean',
    ]


import _stats_quantiles as _quantiles
from _stats_quantiles import _Quartiles

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
    def inner(data, *args, **kwargs):
        data = sorted(data)
        return func(data, *args, **kwargs)
    return inner


def multivariate(func):
    """Decorator to handle multivariate statistics."""
    @functools.wraps(func)
    def inner(xdata, ydata=None):
        xydata = combine_xydata(xdata, ydata)
        return func(xydata)
    return inner


def minmax(*values, **kw):
    """minmax(iterable [, key=func]) -> (minimum, maximum)
    minmax(a, b, c, ... [key=func]) -> (minimum, maximum)

    With a single iterable argument, return a two-tuple of its smallest
    item and largest item. With two or more arguments, return the smallest
    and largest arguments.
    """
    if len(values) == 0:
        raise TypeError('minmax expected at least one argument, but got none')
    elif len(values) == 1:
        values = values[0]
    if isinstance(values, collections.Sequence):
        # For speed, fall back on built-in min and max functions when
        # data is a sequence and can be safely iterated over twice.
        minimum = min(values, **kw)
        maximum = max(values, **kw)
    else:
        # Iterator argument, so fall back on a slow pure-Python solution.
        if list(kw.keys()) not in ([], ['key']):
            raise TypeError('minmax received an unexpected keyword argument')
        key = kw.get('key')
        if key is not None:
            it = ((key(value), value) for value in values)
        else:
            it = ((value, value) for value in values)
        try:
            keyed_min, minimum = next(it)
        except StopIteration:
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


def as_sequence(iterable):
    """Helper function to convert iterable arguments into sequences."""
    if isinstance(iterable, (list, tuple)): return iterable
    else: return list(iterable)


def combine_xydata(xdata, ydata=None):
    """Helper function which combines xdata, ydata to xydata."""
    if ydata is not None:
        # Two argument version is easy.
        return zip(xdata, ydata)
    # The single argument case could be either [x0, x1, x2, ...] or
    # [(x0, y0), (x1, y1), (x2, y2), ...]. We decide which it is by looking
    # at the first item, and treating it as canonical.
    it = iter(xdata)
    try:
        first = next(it)
    except StopIteration:
        # If the iterable is empty, return the original.
        return xdata
    # If we get here, we know we have a single iterable argument with at
    # least one item. Does it look like a sequence of (x,y) values, or like
    # a sequence of x values?
    try:
        len(first)
    except TypeError:
        # Looks like we're dealing with the case [x0, x1, x2, ...]
        first = (first, None)
        tail = ((x, None) for x in it)
        return itertools.chain([first], tail)
    # Looks like [(x0, y0), (x1, y1), (x2, y2), ...]
    # Notice that we DON'T care how many items are in the data points here,
    # we postpone dealing with any mismatches to later.
    return itertools.chain([first], it)


def _validate_int(n):
    # This will raise TypeError, OverflowError (for infinities) or
    # ValueError (for NANs or non-integer numbers).
    if n != int(n):
        raise ValueError('requires integer value')


# === Basic univariate statistics ===


# Measures of central tendency (means and averages)
# -------------------------------------------------


def mean(data):
    """Return the sample arithmetic mean of a sequence of numbers.

    >>> mean([1.0, 2.0, 3.0, 4.0])
    2.5

    The arithmetic mean is the sum of the data divided by the number of data.
    It is commonly called "the average". It is a measure of the central
    location of the data.
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
        # FIXME given that this is a re-implementation of the math.fsum
        # algorithm, is there any advantage to calling math.fsum here?
        # Or should we just call the built-in sum on the partials?
        sumx = math.fsum(partials)
    else:
        sumx = sum(data)  # Not the built-in version.
    if n == 0:
        raise StatsError('mean of empty sequence is not defined')
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
    return math.sqrt(mean(x*x for x in data))


@sorted_data
def median(data, sign=0):
    """Returns the median (middle) value of a sequence of numbers.

    >>> median([3.0, 5.0, 2.0])
    3.0

    The median is the middle data point in a sorted sequence of values. If
    the argument to median is a list, it will be sorted in place, otherwise
    the values will be collected into a sorted list.

    The median is commonly used as an average. It is more robust than the
    mean for data that contains outliers. The median is equivalent to the
    second quartile or the 50th percentile.

    Optional numeric argument sign specifies the behaviour of median in the
    case where there is an even number of elements:

    sign  value returned as median
    ----  ------------------------------------------------------
    0     The mean of the elements on either side of the middle
    < 0   The element just below the middle
    > 0   The element just above the middle

    The default is 0. Except for certain specialist applications, this is
    normally what is expected for the median.
    """
    n = len(data)
    if n == 0:
        raise StatsError('no median for empty iterable')
    m = n//2
    if n%2 == 1:
        # For an odd number of items, there is only one middle element, so
        # we always take that.
        return data[m]
    else:
        # If there are an even number of items, we decide what to do
        # according to sign:
        if sign == 0:
            # Take the mean of the two middle elements.
            return (data[m-1] + data[m])/2
        elif sign < 0:
            # Take the lower middle element.
            return data[m-1]
        elif sign > 0:
            # Take the higher middle element.
            return data[m]
        else:
            raise TypeError('sign is not ordered with respect to zero')


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


def midrange(data):
    """Returns the midrange of a sequence of numbers.

    >>> midrange([2.0, 4.5, 7.5])
    4.75

    The midrange is halfway between the smallest and largest element. It is
    a weak measure of central tendency.
    """
    try:
        L, H = minmax(data)
    except ValueError as e:
        e.args = ('no midrange defined for empty iterables',)
        raise
    return (L + H)/2


def midhinge(data):
    """Return the midhinge of a sequence of numbers.

    >>> midhinge([1, 1, 2, 3, 4, 5, 6, 7, 8, 8])
    4.5

    The midhinge is halfway between the first and second hinges. It is a
    better measure of central tendency than the midrange, and more robust
    than the sample mean (more resistant to outliers).
    """
    H1, _, H2 = hinges(data)
    return (H1 + H2)/2


@sorted_data
def trimean(data):
    """Return Tukey's trimean = (H1 + 2*M + H2)/4 of data


    >>> trimean([1, 1, 3, 5, 7, 9, 10, 14, 18])
    6.75
    >>> trimean([0, 1, 2, 3, 4, 5, 6, 7, 8])
    4.0

    The trimean is equivalent to the average of the median and the midhinge,
    and is considered a better measure of central tendancy than either alone.
    """
    H1, M, H2 = hinges(data)
    return (H1 + 2*M + H2)/4


# Moving averages
# ---------------

def running_average(data):
    """Iterate over data, yielding the running average.

    >>> list(running_average([40, 30, 50, 46, 39, 44]))
    [40, 35.0, 40.0, 41.5, 41.0, 41.5]

    The running average is also known as the cumulative moving average.
    Given data [a, b, c, d, ...] it yields the values:
        a, (a+b)/2, (a+b+c)/3, (a+b+c+d)/4, ...

    that is, the average of the first item, the first two items, the first
    three items, the first four items, ...
    """
    it = iter(data)
    ca = next(it)
    yield ca
    for i, x in enumerate(it, 2):
        ca = (x + (i-1)*ca)/i
        yield ca


def weighted_running_average(data):
    """Iterate over data, yielding a running average with exponentially
    decreasing weights.

    >>> list(weighted_running_average([40, 30, 50, 46, 39, 44]))
    [40, 35.0, 42.5, 44.25, 41.625, 42.8125]

    This running average yields the average between the previous running
    average and the current data point. Given data [a, b, c, d, ...] it
    yields the values:
        a, (a+b)/2, ((a+b)/2 + c)/2, (((a+b)/2 + c)/2 + d)/2, ...

    The values yielded are weighted means where the weight on older points
    decreases exponentially.
    """
    it = iter(data)
    ca = next(it)
    yield ca
    for x in it:
        ca = (ca + x)/2
        yield ca


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


# Quantiles (fractiles) and hinges
# --------------------------------

# Grrr arggh!!! Nobody can agree on how to calculate quantiles.
# Langford (2006) finds no fewer than FIFTEEN methods for calculating
# quartiles (although some are mathematically equivalent to others):
#   http://www.amstat.org/publications/jse/v14n3/langford.html
# Mathword and Dr Math suggest five:
#   http://mathforum.org/library/drmath/view/60969.html
#   http://mathworld.wolfram.com/Quartile.html


@sorted_data
def quartiles(data, method=0):
    """quartiles(data [, method]) -> (Q1, Q2, Q3)

    Return the sample quartiles (Q1, Q2, Q3) for data.

    >>> quartiles([0.5, 2.0, 3.0, 4.0, 5.0, 6.0])
    (2.0, 3.5, 5.0)

    data must be an iterator of numeric values, with at least three items.
    method is an optional value specifying the calculation method used, and
    hence the results. Returns a 3-tuple of the first, second and third
    quartiles (Q1, Q2, Q3). Q2 is not necessarily the median.

    The values returned are such that (approximately) one quarter of the data
    is below Q1, one half below Q2, and three-quarters below Q3. Q1 is
    approximately the median of the first half of the data, and Q3 the median
    of the second half. The exact proportions, and the exact values of the
    cut-offs, depend on the method of calculation. To specify the method of
    calculation, pass a value as the method argument:

    Method  Description
    ======  ================================================================
    0       Tukey's hinges; median is included in the two halves
    1       Moore and McCabe's method; median is excluded in the two halves
    2       Method recommended by Mendenhall and Sincich
    3       Method used by Minitab software
    4       Method recommended by Freund and Perles
    5       Langford's CDF method

    * Method 0 (the default) is equivalent to Tukey's hinges (H1, M, H2).
    * Method 1 is used by Texas Instruments calculators, model TI-85 and up.
    * Method 2 ensures that the values returned are always data points,
      which makes it suitable for ordinal data.
    * Methods 3 and 4 use linear iterpolation between items.
    * Method 4 is used by Microsoft Excel and OpenOffice.

    Case-insensitive named aliases are also supported for methods: you can
    examine quartiles.aliases for a mapping of names to method numbers.
    """
    # Select a method.
    if isinstance(method, str):
        key = quartiles.aliases.get(method.lower())
    else:
        key = method
    func = _Quartiles.QUARTILE_MAP.get(key)
    if func is None:
        raise StatsError('unrecognised method selector `%s`' % method)
    # Return the quartiles using that method.
    n = len(data)
    if n < 3:
        raise StatsError('need at least 3 items to split data into quartiles')
    return func(data)

quartiles.aliases = _Quartiles.QUARTILE_ALIASES


def hinges(data):
    """Return Tukey's hinges H1, M, H2 from data.

    >>> hinges([2, 4, 6, 8, 10, 12, 14, 16, 18])
    (6, 10, 14)

    This is equivalent to quartiles() called with method 0.
    """
    return quartiles(data, 0)


# Quantiles (fractiles) are just as confused as quartiles. The statistics
# language R offers nine different methods for calculating quantiles. We
# support ???? of them.

@sorted_data
def quantile(data, p, method=0):
    """quantile(data, p [, method]) -> value

    Return the p-quantile which is some fraction p of the way into data.

    >>> quantile([2.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0.75)
    4.75

    data must be an iterator of numeric values, with at least two items.
    p must be a number between 0 and 1 inclusive. method is an optional value
    specifying the calculation method used, and hence the result.

    The result returned by quantile is the data point, or the interpolated
    data point, such that a fraction p of the data is less than that value.

    Method  Description
    ======  ================================================================
    0       Langford's method #4 based on cumulative distribution function
    1-9     Types 1-9 from R (#2 is equivalent to 0)
    10      ...

    See the R manual for further details:
    http://stat.ethz.ch/R-manual/R-devel/library/stats/html/quantile.html

    Case-insensitive named aliases are also supported for methods: you can
    examine quantiles.aliases for a mapping of names to method numbers.
    """
    # Select a method.
    if isinstance(method, str):
        key = quantile.aliases.get(method.lower())
    else:
        key = method
    func = _quantiles.QUANTILE_MAP.get(key)
    if func is None:
        raise StatsError('unrecognised method selector `%s`' % method)

    if not 0.0 <= p <= 1.0:
        raise StatsError('quantile argument must be between 0.0 and 1.0')
    n = len(data)
    if n < 2:
        raise StatsError('need at least 2 items to split data into quantiles')
    x = p*(n-1)
    m = int(x)
    p = x - m
    if p:
        a = data[m]
        delta = data[m+1] - a
        return a + p*delta
    else:
        return data[m]

quantile.aliases = _quantiles.QUANTILE_ALIASES


def decile(data, d, method=0):
    """Return the dth decile of data, for integer d between 0 and 10.

    See function quantile for details about the optional argument method.
    """
    _validate_int(d)
    if not 0 <= d <= 10:
        raise ValueError('decile argument d must be between 0 and 10')
    return quantile(data, d/10, method)


def percentile(data, p, method=0):
    """Return the pth percentile of data, for integer p between 0 and 100.

    See function quantile for details about the optional argument method.
    """
    _validate_int(p)
    if not 0 <= p <= 100:
        raise ValueError('percentile argument p must be between 0 and 100')
    return quantile(data, p/100, method)


def boxwhiskerplot(*args, **kwargs):
    pass  # placeholder


# Measures of spread (dispersion or variability)
# ----------------------------------------------

def pvariance(data, m=None):
    """pvariance(data [, m]) -> population variance of data.

    >>> pvariance([0.25, 0.5, 1.25, 1.25,
    ...           1.75, 2.75, 3.5])  #doctest: +ELLIPSIS
    1.17602040816...

    If you know the population mean, or an estimate of it, then you can pass
    the mean as the optional argument m. See also pstdev.

    If data represents a statistical sample rather than the entire
    population, you should use variance instead.
    """
    n, s = _var(data, m)
    if n < 1:
        raise StatsError('population variance or standard deviation'
        ' requires at least one data point')
    return s/n


def variance(data, m=None):
    """variance(data [, m]) -> sample variance of data.

    >>> variance([0.25, 0.5, 1.25, 1.25,
    ...           1.75, 2.75, 3.5])  #doctest: +ELLIPSIS
    1.37202380952...

    If you know the population mean, or an estimate of it, then you can pass
    the mean as the optional argument m. See also stdev.

    If data represents the entire population, and not just a sample, then
    you should use pvariance instead.
    """
    n, s = _var(data, m)
    if n < 2:
        raise StatsError('sample variance or standard deviation'
        ' requires at least two data points')
    return s/(n-1)


def _var(data, m):
    """Helper function for calculating variance directly."""
    if m is None:
        # Two pass algorithm.
        data = as_sequence(data)
        m = mean(data)
    # Fast path for sequences.
    try:
        n = len(data)
    except TypeError:
        # Slow path for iterables without a len.
        ap = add_partial
        partials = []
        for n, x in enumerate(data, 1):
            ap((x-m)**2, partials)
        total = sum(partials)
    else:
        total = sum((x-m)**2 for x in data)
    return (n, total)
    # FIXME this may not be accurate enough, a more accurate algorithm
    # is the compensated version:
    # sum2 = sum(x-m)**2) as above
    # sumc = sum(x-m)
    # return (sum2 - sumc**2/n), n


def pstdev(data, m=None):
    """pstdev(data [, m]) -> population standard deviation of data.

    >>> pstdev([1.5, 2.5, 2.5, 2.75, 3.25, 4.75])  #doctest: +ELLIPSIS
    0.986893273527...

    If you know the true population mean by some other means, then you can
    pass that as the optional argument m:

    >>> pstdev([1.5, 2.5, 2.5, 2.75, 3.25, 4.75], 2.875)  #doctest: +ELLIPSIS
    0.986893273527...

    The reliablity of the result as an estimate for the true standard
    deviation depends on the estimate for the mean given. If m is not given,
    or is None, the sample mean of the data will be used.

    If data represents a statistical sample rather than the entire
    population, you should use stdev instead.
    """
    return math.sqrt(pvariance(data, m))


def stdev(data, m=None):
    """stdev(data [, m]) -> sample standard deviation of data.

    >>> stdev([1.5, 2.5, 2.5, 2.75, 3.25, 4.75])  #doctest: +ELLIPSIS
    1.08108741552...

    If you know the population mean, or an estimate of it, then you can pass
    the mean as the optional argument m:

    >>> stdev([1.5, 2.5, 2.75, 2.75, 3.25, 4.25], 3)  #doctest: +ELLIPSIS
    0.921954445729...

    The reliablity of the result as an estimate for the true standard
    deviation depends on the estimate for the mean given. If m is not given,
    or is None, the sample mean of the data will be used.

    If data represents the entire population, and not just a sample, then
    you should use pstdev instead.
    """
    return math.sqrt(variance(data, m))


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
    # passes.
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


def range(data):
    """Return the statistical range of data.

    >>> range([1.0, 3.5, 7.5, 2.0, 0.25])
    7.25

    The range is the difference between the smallest and largest element. It
    is a weak measure of statistical variability.
    """
    try:
        a, b = minmax(data)
    except ValueError as e:
        e.args = ('no range defined for empty iterables',)
        raise
    return b - a


def iqr(data, method=0):
    """Returns the Inter-Quartile Range of a sequence of numbers.

    >>> iqr([0.5, 2.25, 3.0, 4.5, 5.5, 6.5])
    3.25

    The IQR is the difference between the first and third quartile. The
    optional argument method (defaulting to 0) is used to select the
    algorithm for calculating the quartiles. See the quartile function for
    further details.

    The default IQR (method 0) is equivalent to Tukey's H-spread.
    """
    q1, _, q3 = quartiles(data, method)
    return q3 - q1


def average_deviation(data, m=None):
    """average_deviation(data [, m]) -> average absolute deviation of data.

    data = iterable of data values
    m (optional) = measure of central tendency for data.

    m is usually chosen to be the mean or median. If m is not given, or
    is None, the mean is calculated from data and that value is used.
    """
    if m is None:
        data = as_sequence(data)
        m = mean(data)
    # Try a fast path for sequence data.
    try:
        n = len(data)
    except TypeError:
        # Slow path for iterables without a len.
        ap = add_partial
        partials = []
        n = 0
        for x in xdata:
            n += 1
            ap(abs(x - m), partials)
        total = sum(partials)
    else:
        total = sum(abs(x-m))
    if n < 1:
        raise StatsError('average deviation requires at least 1 data point')
    return total/n


def median_average_deviation(data, m=None, sign=0, scale=1):
    """Compute the median absolute deviation (MAD) of data.

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
    'uniform'   Apply a scale factor of 1.1547, applicable to data from a
                uniform distribution.

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
    }


# Other moments of the data
# -------------------------

def quartile_skewness(q1, q2, q3):
    """Return the quartile skewness coefficient, or Bowley skewness, from
    the three quartiles q1, q2 (median), q3.

    >>> quartile_skewness(1, 2, 5)
    0.5
    >>> quartile_skewness(1, 4, 5)
    -0.5

    """
    return (q3 + q1 - 2*q2)/(q3 - q1)


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

    Returns the degree to which the data is skewed to the left or the right
    of the mean.

    If you know one or both of the population mean and standard deviation,
    or estimates of them, then you can pass the mean as optional argument m
    and the standard deviation as s.

    >>> skewness([1.25, 1.5, 1.5, 1.75, 1.75, 2.5, 2.75, 4.5])
    ... #doctest: +ELLIPSIS
    1.12521290135...

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
    """
    if m is None or s is None:
        data = as_sequence(data)
        if m is None: m = mean(data)
        if s is None: s = stdev(data, m)
    # Try fast path.
    try:
        n = len(data)
    except TypeError:
        # Slow path for iterables without len.
        ap = add_partial
        partials = []
        n = 0
        for x in xdata:
            n += 1
            ap(((x - m)/s)**3, partials)
        total = sum(partials)
    else:
        total = sum(((x-m)/s)**3 for x in data)
    return total/n


def kurtosis(data, m=None, s=None):
    """kurtosis(data [,m [,s]]) -> sample kurtosis of data.

    Returns the excess kurtosis (degree of peakedness) of the data, relative
    to the kurtosis of the normal distribution. To convert to kurtosis proper,
    add 3 to the result.

        WARNING: As the kurtosis depends on the fourth power of the
        data points, it can suffer from significant round-off error
        and should be used with caution.

    If you know one or both of the population mean and standard deviation,
    or estimates of them, then you can pass the mean as optional argument m
    and the standard deviation as s.

    >>> kurtosis([1.25, 1.5, 1.5, 1.75, 1.75, 2.5, 2.75, 4.5])
    ... #doctest: +ELLIPSIS
    -0.1063790369...

    The reliablity of the result as an estimate for the kurtosis depends on
    the estimated mean and standard deviation given. If m or s are not given,
    or are None, they are estimated from the data.

    The higher the kurtosis, the sharper the peak. Higher kurtosis means more
    of the variance is the result of infrequent extreme deviations, as opposed
    to frequent modestly sized deviations.
    """
    if m is None or s is None:
        data = as_sequence(data)
        if m is None: m = mean(data)
        if s is None: s = stdev(data, m)
    # Try fast path.
    try:
        n = len(data)
    except TypeError:
        # Slow path for iterables without len.
        ap = add_partial
        partials = []
        n = 0
        for x in xdata:
            n += 1
            ap(((x - m)/s)**4, partials)
        total = sum(partials)
    else:
        total = sum(((x-m)/s)**4 for x in data)
    k = total/n - 3
    assert k >= -2
    return k


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




# === Simple multivariate statistics ===


def qcorr(xdata, ydata=None):
    """Return the Q correlation coefficient of (x, y) data.

    if ydata is None or not given, then xdata must be an iterable of (x, y)
    pairs. Otherwise, both xdata and ydata must be iterables of values, which
    will be truncated to the shorter of the two.

    qcorr(xydata) -> float
    qcorr(xdata, ydata) -> float

    The Q correlation can be found by drawing a scatter graph of the points,
    diving the graph into four quadrants by marking the medians of the X
    and Y values, and then counting the points in each quadrant. Points on
    the median lines are skipped.

    The Q correlation coefficient is +1 in the case of a perfect positive
    correlation (i.e. an increasing linear relationship):

    >>> qcorr([1, 2, 3, 4, 5], [3, 5, 7, 9, 11])
    1.0

    -1 in the case of a perfect anti-correlation (i.e. a decreasing linear
    relationship):

    >>> qcorr([(1, 10), (2, 8), (3, 6), (4, 4), (5, 2)])
    -1.0

    and some value between -1 and 1 in all other cases, indicating the degree
    of linear dependence between the variables.
    """
    if ydata is None:
        xydata = list(xdata)
    else:
        xydata = list(zip(xdata, ydata))
    del xdata, ydata
    n = len(xydata)
    xmed = median(x for x,y in xydata)
    ymed = median(y for x,y in xydata)
    # Traditionally, we count the values in each quadrant, but in fact we
    # really only need to count the diagonals: quadrants 1 and 3 together,
    # and quadrants 2 and 4 together.
    quad13 = quad24 = skipped = 0
    for x,y in xydata:
        if x > xmed:
            if y > ymed:  quad13 += 1
            elif y < ymed:  quad24 += 1
            else:  skipped += 1
        elif x < xmed:
            if y > ymed:  quad24 += 1
            elif y < ymed:  quad13 += 1
            else:  skipped += 1
        else:
            skipped += 1
    assert quad13 + quad24 + skipped == n
    q = (quad13 - quad24)/(n - skipped)
    assert -1.0 <= q <= 1.0
    return q


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


def corr1(xdata, ydata):
    """Calculate an estimate of the Pearson's correlation coefficient with
    a single pass over the data."""
    sum_sq_x = 0
    sum_sq_y = 0
    sum_coproduct = 0
    mean_x = next(xdata)
    mean_y = next(ydata)
    for i,x,y in zip(itertools.count(2), xdata, ydata):
        sweep = (i-1)/i
        delta_x = x - mean_x
        delta_y = y - mean_y
        sum_sq_x += sweep*delta_x**2
        sum_sq_y += sweep*(delta_y**2)
        sum_coproduct += sweep*(delta_x*delta_y)
        mean_x += delta_x/i
        mean_y += delta_y/i
    pop_sd_x = sqrt(sum_sq_x)
    pop_sd_y = sqrt(sum_sq_y)
    cov_x_y = sum_coproduct
    r = cov_x_y/(pop_sd_x*pop_sd_y)
    assert -1.0 <= r <= 1.0
    return r


def pcov(xdata, ydata=None):
    """Return the population covariance between (x, y) data.

    >>> print(pcov([0.75, 1.5, 2.5, 2.75, 2.75], [0.25, 1.1, 2.8, 2.95, 3.25]))
    0.934
    >>> print(pcov([(0.1, 2.3), (0.5, 2.7), (1.2, 3.1), (1.7, 2.9)]))
    0.15125

    """
    t = xysums(xdata, ydata)
    return t.Sxy/(t.n**2)


def cov(xdata, ydata=None):
    """Return the sample covariance between (x, y) data.

    >>> cov([(0.1, 2.3), (0.5, 2.7), (1.2, 3.1),
    ...      (1.7, 2.9)])  #doctest: +ELLIPSIS
    0.201666666666...

    >>> print(cov([0.75, 1.5, 2.5, 2.75, 2.75], [0.25, 1.1, 2.8, 2.95, 3.25]))
    1.1675
    >>> cov([(0.1, 2.3), (0.5, 2.7), (1.2, 3.1), (1.7, 2.9)])
    ... #doctest: +ELLIPSIS
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
    cancellation, e.g. sum([1, 1e100, 1, -1e100] * 10000) returns zero
    instead of the correct value of 20000. This function avoids that error:

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


@multivariate
def Sxx(xydata):
    """Return Sxx = n*sum(x**2) - sum(x)**2 from (x,y) data or x data alone.

    Returns Sxx from either a single iterable or a pair of iterables.

    If given a single iterable argument, it must be either the (x,y) values,
    in which case the y values are ignored, or the x values alone:

    >>> Sxx([(1, 2), (3, 4), (5, 8)])
    24.0
    >>> Sxx([1, 3, 5])
    24.0

    In the two argument form, Sxx(xdata, ydata), the second argument ydata
    is ignored except that the data is truncated at the shorter of the
    two arguments:

    >>> Sxx([1, 3, 5, 7, 9], [2, 4, 8])
    24.0

    """
    n, s = _var((x for (x, y) in xydata), None)
    return s*n


@multivariate
def Syy(xydata):
    """Return Syy = n*sum(y**2) - sum(y)**2 from (x,y) data or y data alone.

    Returns Syy from either a single iterable or a pair of iterables.

    If given a single iterable argument, it must be either the (x,y) values,
    in which case the x values are ignored, or the y values alone:

    >>> Syy([(1, 2), (3, 4), (5, 8)])
    56.0
    >>> Syy([2, 4, 8])
    56.0

    In the two argument form, Syy(xdata, ydata), the first argument xdata
    is ignored except that the data is truncated at the shorter of the
    two arguments:

    >>> Syy([1, 3, 5], [2, 4, 8, 16, 32])
    56.0

    """
    # We expect (x,y) points, but if the caller passed a single iterable
    # ydata as argument, it gets mistaken as xdata with the y values all
    # set to None. (See the combine_xydata function.) We have to detect
    # that and swap the values around.
    try:
        first = next(xydata)
    except StopIteration:
        pass  # Postpone dealing with this.
    else:
        if len(first) == 2 and first[1] is None:
            # Swap the elements around.
            first = (first[1], first[0])
            xydata = ((x, y) for (y, x) in xydata)
        # Re-insert the first element back into the data stream.
        xydata = itertools.chain([first], xydata)
    n, s = _var((y for (x, y) in xydata), None)
    return s*n


@multivariate
def Sxy(xydata):
    """Return Sxy = n*sum(x*y) - sum(x)*sum(y) from (x,y) data.

    Returns Sxy from either a single iterable or a pair of iterables.

    If given a single iterable argument, it must be the (x,y) values:

    >>> Sxy([(1, 2), (3, 4), (5, 8)])
    36.0

    In the two argument form, Sxx(xdata, ydata), data is truncated at the
    shorter of the two arguments:

    >>> Sxy([1, 3, 5, 7, 9], [2, 4, 8])
    36.0

    """
    n = 0
    sumx, sumy, sumxy = [], [], []
    ap = add_partial
    fsum = math.fsum
    for x, y in xydata:
        n += 1
        ap(x, sumx)
        ap(y, sumy)
        ap(x*y, sumxy)
    return n*fsum(sumxy) - fsum(sumx)*fsum(sumy)


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
    radians, cos, sin = math.radians, math.cos, math.sin
    ap = add_partial
    if deg:
        data = (radians(theta) for theta in data)
    n, cosines, sines = 0, [], []
    for n, theta in enumerate(data, 1):
        ap(cos(theta), cosines)
        ap(sin(theta), sines)
    if n == 0:
        raise StatsError('circular mean of empty sequence is not defined')
    x = math.fsum(cosines)/n
    y = math.fsum(sines)/n
    theta = math.atan2(y, x)  # Note the order is swapped.
    if deg:
        theta = math.degrees(theta)
    return theta



if __name__ == '__main__':
    import doctest
    doctest.testmod()

