#!/usr/bin/env python3

##  Copyright (c) 2011 Steven D'Aprano.
##  See __init__.py for the licence terms for this software.

"""
Order statistics.

"""
# TODO: investigate finding median and other fractiles without sorting,
# e.g. QuickSelect, ranking, etc. See:
# http://mail.gnome.org/archives/gnumeric-list/2007-February/msg00023.html
# http://mail.gnome.org/archives/gnumeric-list/2007-February/msg00041.html


__all__ = [
    'median', 'midrange', 'midhinge', 'trimean',
    'range', 'iqr', 
    'quartile_skewness',
    'hinges', 'quartiles', 'quantile', 'decile', 'percentile',
    'QUARTILE_DEFAULT', 'QUANTILE_DEFAULT',
    ]


import functools
import math

from stats import StatsError as _StatsError

import stats.utils
#from stats.utils import sorted_data
#from stats.utils import StatsError, minmax, _round, _UP, _DOWN, _EVEN



# === Global variables ===

# Default schemes to use for order statistics:
QUARTILE_DEFAULT = 1
QUANTILE_DEFAULT = 1



# === Utilities ===

def sorted_data(func):
    """Decorator to sort data passed to stats functions."""
    @functools.wraps(func)
    def inner(data, *args, **kwargs):
        data = sorted(data)
        return func(data, *args, **kwargs)
    return inner


# === Private utilities ===

def _interpolate(data, x):
    i, f = math.floor(x), x%1
    if f:
        a, b = data[i], data[i+1]
        return a + f*(b-a)
    else:
        return data[i]


def _round_up(x):
    """Round non-negative x, rounding ties up."""
    assert x >= 0.0
    n, f = int(x), x%1
    if f >= 0.5:
        return n+1
    else:
        return n


def _round_down(x):
    """Round non-negative x, rounding ties down."""
    assert x >= 0.0
    n, f = int(x), x%1
    if f > 0.5:
        return n+1
    else:
        return n


def _round_even(x):
    """Round non-negative x, using Banker's rounding to even for ties."""
    assert x >= 0.0
    n, f = int(x), x%1
    if f > 0.5:
        return n+1
    elif f < 0.5:
        return n
    else:
        if n%2:
            # n is odd, so round up to even.
            return n+1
        else:
            # n is even, so round down.
            return n


# === Order statistics ===

# Measures of central tendency (averages)
# ---------------------------------------


def median(data, sign=0):
    """Returns the median (middle) value of an iterable of numbers.

    >>> median([3.0, 5.0, 2.0])
    3.0

    The median is the middle data point in a sorted sequence of values, and
    is commonly used as an average. It is more robust than the mean for data
    that contains outliers -- if your data contains a few items that are
    extremely small, or extremely large, compared to the rest of the data,
    the median will be more representative of the data than the mean.

    The median is equivalent to the second quartile or the 50th percentile.

    If there are an odd number of elements in the data, the median is
    always the middle one. If there are an even number of elements, there
    is no middle element, and the value returned by ``median`` depends on
    the optional numeric argument ``sign``:

    sign  value returned as median
    ----  -----------------------------------------------------------------
    = 0   The mean of the elements on either side of the middle (default).
    < 0   The element just below the middle ("low median").
    > 0   The element just above the middle ("high median").

    Normally you will want to stick with the default.
    """
    data = sorted(data)
    n = len(data)
    if n == 0:
        raise _StatsError('no median for empty iterable')
    m = n//2
    if n%2 == 1:
        # For an odd number of items, there is only one middle element, so
        # we always take that.
        return data[m]
    else:
        # If there are an even number of items, we decide what to do
        # according to sign.
        if sign == 0:
            # Take the mean of the two middle elements.
            return (data[m-1] + data[m])/2
        elif sign < 0:
            # Low median: take the lower middle element.
            return data[m-1]
        elif sign > 0:
            # High median: take the higher middle element.
            return data[m]
        else:
            # Unordered numeric value for sign. Probably a NAN.
            raise ValueError('sign is not ordered with respect to zero')


def midrange(data):
    """Returns the midrange of a sequence of numbers.

    >>> midrange([2.0, 4.5, 7.5])
    4.75

    The midrange is halfway between the smallest and largest element. It is
    a weak measure of central tendency.
    """
    try:
        L, H = stats.utils.minmax(data)
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


# Measures of spread (dispersion or variability)
# ----------------------------------------------

def range(data):
    """Return the statistical range of data.

    >>> range([1.0, 3.5, 7.5, 2.0, 0.25])
    7.25

    The range is the difference between the smallest and largest element. It
    is a weak measure of statistical variability.
    """
    try:
        a, b = stats.utils.minmax(data)
    except ValueError as e:
        e.args = ('no range defined for empty iterables',)
        raise
    return b - a


def iqr(data, scheme=None):
    """Returns the Inter-Quartile Range of a sequence of numbers.

    >>> iqr([0.5, 2.25, 3.0, 4.5, 5.5, 6.5])
    3.25

    The IQR is the difference between the first and third quartile. The
    optional argument scheme is used to select the algorithm for calculating
    the quartiles. The default scheme is taken from the global variable
    QUARTILE_DEFAULT. See the quartile function for further details.

    The IQR with scheme 1 is equivalent to Tukey's H-spread.
    """
    q1, _, q3 = quartiles(data, scheme)
    return q3 - q1


# Other moments of the data
# -------------------------

def quartile_skewness(q1, q2, q3):
    """Return the quartile skewness coefficient, or Bowley skewness, from
    the three quartiles q1, q2, q3.

    >>> quartile_skewness(1, 2, 5)
    0.5
    >>> quartile_skewness(1, 4, 5)
    -0.5

    """
    if not q1 <= q2 <= q3:
        raise _StatsError('quartiles must be ordered q1 <= q2 <= q3')
    if q1 == q2 == q3:
        return float('nan')
    skew = (q3 + q1 - 2*q2)/(q3 - q1)
    assert -1.0 <= skew <= 1.0
    return skew


# Fractiles: hinges, quartiles and quantiles
# ------------------------------------------

# Grrr arggh!!! Nobody can agree on how to calculate order statistics.
# Langford (2006) finds no fewer than FIFTEEN methods for calculating
# quartiles (although some are mathematically equivalent to others):
#   http://www.amstat.org/publications/jse/v14n3/langford.html
#
# Mathword and Dr Math suggest five:
#   http://mathforum.org/library/drmath/view/60969.html
#   http://mathworld.wolfram.com/Quartile.html
#
# Even calculating the median is open to disagreement. There has also been
# some discussion on the Gnumeric spreadsheet list:
#   http://mail.gnome.org/archives/gnumeric-list/2007-February/msg00041.html
# with this quote from J Nash summarising the whole mess:
#
#       Ultimately, this question boils down to where to cut to
#       divide 4 candies among 5 children. No matter what you do,
#       things get ugly.
#
# Quantiles (fractiles) and percentiles also have a plethora of calculation
# methods. R (and presumably S) include nine different methods for quantiles.
# Mathematica uses a parameterized quantile function capable of matching
# eight of those nine methods. Wikipedia lists a tenth method. There are
# probably others I don't know of. And then there are grouped and weighted
# data to deal with too :(


# Private functions for fractiles:

class _Quartiles:
    """Private namespace for quartile calculation methods.

    ALL functions and attributes in this namespace class are private and
    subject to change without notice.
    """
    def __new__(cls):
        raise RuntimeError('namespace, do not initialise')

    # Implementation notes
    # --------------------
    #
    # All the following functions assume that data is a sorted list.

    def inclusive(data):
        """Return sample quartiles using Tukey's method.

        Q1 and Q3 are calculated as the medians of the two halves of the data,
        where the median Q2 is included in both halves. This is equivalent to
        Tukey's hinges H1, M, H2.
        """
        n = len(data)
        i = (n+1)//4
        m = n//2
        if n%4 in (0, 3):
            q1 = (data[i] + data[i-1])/2
            q3 = (data[-i-1] + data[-i])/2
        else:
            q1 = data[i]
            q3 = data[-i-1]
        if n%2 == 0:
            q2 = (data[m-1] + data[m])/2
        else:
            q2 = data[m]
        return (q1, q2, q3)

    def exclusive(data):
        """Return sample quartiles using Moore and McCabe's method.

        Q1 and Q3 are calculated as the medians of the two halves of the data,
        where the median Q2 is excluded from both halves.

        This is the method used by Texas Instruments model TI-85 calculator.
        """
        n = len(data)
        i = n//4
        m = n//2
        if n%4 in (0, 1):
            q1 = (data[i] + data[i-1])/2
            q3 = (data[-i-1] + data[-i])/2
        else:
            q1 = data[i]
            q3 = data[-i-1]
        if n%2 == 0:
            q2 = (data[m-1] + data[m])/2
        else:
            q2 = data[m]
        return (q1, q2, q3)

    def ms(data):
        """Return sample quartiles using Mendenhall and Sincich's method."""
        # Perform index calculations using 1-based counting, and adjust for
        # 0-based at the very end.
        n = len(data)
        M = _round_even((n+1)/2)
        L = _round_up((n+1)/4)
        U = n+1-L
        assert U == _round_down(3*(n+1)/4)
        return (data[L-1], data[M-1], data[U-1])

    def minitab(data):
        """Return sample quartiles using the method used by Minitab."""
        # Perform index calculations using 1-based counting, and adjust for
        # 0-based at the very end.
        n = len(data)
        M = (n+1)/2
        L = (n+1)/4
        U = n+1-L
        assert U == 3*(n+1)/4
        return (
                _interpolate(data, L-1),
                _interpolate(data, M-1),
                _interpolate(data, U-1)
                )

    def excel(data):
        """Return sample quartiles using Freund and Perles' method.

        This is also the method used by Excel and OpenOffice.
        """
        # Perform index calculations using 1-based counting, and adjust for
        # 0-based at the very end.
        n = len(data)
        M = (n+1)/2
        L = (n+3)/4
        U = (3*n+1)/4
        return (
                _interpolate(data, L-1),
                _interpolate(data, M-1),
                _interpolate(data, U-1)
                )

    def langford(data):
        """Langford's recommended method for calculating quartiles based on
        the cumulative distribution function (CDF).
        """
        n = len(data)
        m = n//2
        i, r = divmod(n, 4)
        if r == 0:
            q1 = (data[i] + data[i-1])/2
            q2 = (data[m-1] + data[m])/2
            q3 = (data[-i-1] + data[-i])/2
        elif r in (1, 3):
            q1 = data[i]
            q2 = data[m]
            q3 = data[-i-1]
        else:  # r == 2
            q1 = data[i]
            q2 = (data[m-1] + data[m])/2
            q3 = data[-i-1]
        return (q1, q2, q3)

    # Numeric method selectors for quartiles:
    QUARTILE_MAP = {
        1: inclusive,
        2: exclusive,
        3: ms,
        4: minitab,
        5: excel,
        6: langford,
        }
        # Note: if you modify this, you must also update the docstring for
        # the quartiles function.

    # Lowercase aliases for the numeric method selectors for quartiles:
    QUARTILE_ALIASES = {
        'cdf': 6,
        'excel': 5,
        'exclusive': 2,
        'f&p': 5,
        'hinges': 1,
        'inclusive': 1,
        'langford': 6,
        'm&m': 2,
        'm&s': 3,
        'minitab': 4,
        'openoffice': 5,
        'ti-85': 2,
        'tukey': 1,
        }
    assert all(alias==alias.lower() for alias in QUARTILE_ALIASES)
# End of private _Quartiles namespace.


class _Quantiles:
    """Private namespace for quantile calculation methods.

    ALL functions and attributes in this namespace class are private and
    subject to change without notice.
    """
    def __new__(cls):
        raise RuntimeError('namespace, do not instantiate')

    # The functions r1...r9 implement R's quartile types 1...9 respectively.
    # Except for r2, they are also equivalent to Mathematica's parametrized
    # quantile function: http://mathworld.wolfram.com/Quantile.html

    # Implementation notes
    # --------------------
    #
    # * The usual formulae for quartiles use 1-based indexes.
    # * Each of the functions r1...r9 assume that data is a sorted sequence,
    #   and that p is a fraction 0 <= p <= 1.

    def r1(data, p):
        h = len(data)*p + 0.5
        i = max(1, math.ceil(h - 0.5))
        assert 1 <= i <= len(data)
        return data[i-1]

    def r2(data, p):
        """Langford's Method #4 for calculating general quantiles using the
        cumulative distribution function (CDF); this is also R's method 2 and
        SAS' method 5.
        """
        n = len(data)
        h = n*p + 0.5
        i = max(1, math.ceil(h - 0.5))
        j = min(n, math.floor(h + 0.5))
        assert 1 <= i <= j <= n
        return (data[i-1] + data[j-1])/2

    def r3(data, p):
        h = len(data)*p
        i = max(1, round(h))
        assert 1 <= i <= len(data)
        return data[i-1]

    def r4(data, p):
        n = len(data)
        if p < 1/n: return data[0]
        elif p == 1.0: return data[-1]
        else: return _interpolate(data, n*p - 1)

    def r5(data, p):
        n = len(data)
        if p < 1/(2*n): return data[0]
        elif p >= (n-0.5)/n: return data[-1]
        h = n*p + 0.5
        return _interpolate(data, h-1)

    def r6(data, p):
        n = len(data)
        if p < 1/(n+1): return data[0]
        elif p >= n/(n+1): return data[-1]
        h = (n+1)*p
        return _interpolate(data, h-1)

    def r7(data, p):
        n = len(data)
        if p == 1: return data[-1]
        h = (n-1)*p + 1
        return _interpolate(data, h-1)

    def r8(data, p):
        n = len(data)
        h = (n + 1/3)*p + 1/3
        h = max(1, min(h, n))
        return _interpolate(data, h-1)

    def r9(data, p):
        n = len(data)
        h = (n + 0.25)*p + 3/8
        h = max(1, min(h, n))
        return _interpolate(data, h-1)

    def qlsd(data, p):
        # This gives the quantile with the least expected square deviation.
        # See http://en.wikipedia.org/wiki/Quantiles
        n = len(data)
        h = (n + 2)*p - 0.5
        h = max(1, min(h, n))
        return _interpolate(data, h-1)

    # Numeric method selectors for quantiles. Numbers 1-9 MUST match the R
    # calculation methods with the same number.
    QUANTILE_MAP = {
        1: r1,
        2: r2,
        3: r3,
        4: r4,
        5: r5,
        6: r6,
        7: r7,
        8: r8,
        9: r9,
        10: qlsd,
        }
        # Note: if you add any additional methods to this, you must also
        # update the docstring for the quantiles function.

    # Lowercase aliases for quantile schemes:
    QUANTILE_ALIASES = {
        'cdf': 2,
        'excel': 7,
        'h&f': 8,
        'hyndman': 8,
        'matlab': 5,
        'minitab': 6,
        'sas-1': 4,
        'sas-2': 3,
        'sas-3': 1,
        'sas-4': 6,
        'sas-5': 2,
        }
    assert all(alias==alias.lower() for alias in QUANTILE_ALIASES)
# End of private _Quantiles namespace.


def _parametrized_quantile(parameters, data, p):
    """_parameterized_quantile(parameters, data, p) -> value

    Private function calculating a parameterized version of quantile,
    equivalent to the Mathematica Quantile() function.

    data is assumed to be sorted and with at least two items; p is assumed
    to be between 0 and 1 inclusive. If either of these assumptions are
    violated, the behaviour of this function is undefined.

    >>> from builtins import range; data = range(1, 21)
    >>> _parametrized_quantile((0, 0, 1, 0), data, 0.3)
    6.0
    >>> _parametrized_quantile((1/2, 0, 0, 1), data, 0.3)
    6.5

        WARNING: While this function will accept arbitrary numberic
        values for the parameters, not all such combinations are
        meaningful:

        >>> _parametrized_quantile((1, 1, 1, 1), [1, 2], 0.3)
        2.9

    """
    # More details here:
    # http://reference.wolfram.com/mathematica/ref/Quantile.html
    # http://mathworld.wolfram.com/Quantile.html
    a, b, c, d = parameters
    n = len(data)
    h = a + (n+b)*p
    f = h % 1
    i = max(1, min(math.floor(h), n))
    j = max(1, min(math.ceil(h), n))
    x = data[i-1]
    y = data[j-1]
    return x + (y - x)*(c + d*f)


# Public functions for fractiles:

def hinges(data):
    """Return Tukey's hinges H1, M, H2 from data.

    >>> hinges([2, 4, 6, 8, 10, 12, 14, 16, 18])
    (6, 10, 14)

    If the data has length N of the form 4n+5 (e.g. 5, 9, 13, 17...) then
    the hinges can be visualised by writing out the ordered data in the
    shape of a W, where each limb of the W is equal is length. For example,
    the data (A,B,C,...) with N=9 would be written out like this:

        A       E       I
          B   D   F   H
            C       G

    and the hinges would be C, E and G.

    This is equivalent to quartiles() called with scheme=1.
    """
    return quartiles(data, scheme=1)


def quartiles(data, scheme=None):
    """quartiles(data [, scheme]) -> (Q1, Q2, Q3)

    Return the sample quartiles (Q1, Q2, Q3) for data, where one quarter of
    the data is below Q1, two quarters below Q2, and three quarters below Q3.
    data must be an iterable of numeric values, with at least three items.

    >>> quartiles([0.5, 2.0, 3.0, 4.0, 5.0, 6.0])
    (2.0, 3.5, 5.0)

    In general, data sets don't divide evenly into four equal sets, and so
    calculating quartiles requires a method for splitting data points. The
    optional argument scheme specifies the calculation method used. The
    exact values returned as Q1, Q2 and Q3 will depend on the method.

    scheme  Description
    ======  =================================================================
    1       Tukey's hinges method; median is included in the two halves
    2       Moore and McCabe's method; median is excluded from the two halves
    3       Method recommended by Mendenhall and Sincich
    4       Method used by Minitab software
    5       Method recommended by Freund and Perles
    6       Langford's CDF method

    Notes:

        (a) If scheme is missing or None, the default is taken from the
            global variable QUARTILE_DEFAULT (set to 1 by default).
        (b) Scheme 1 is equivalent to Tukey's hinges (H1, M, H2).
        (c) Scheme 2 is used by Texas Instruments calculators starting with
            model TI-85.
        (d) Scheme 3 ensures that the values returned are always data points.
        (e) Schemes 4 and 5 use linear interpolation between items.
        (f) For compatibility with Microsoft Excel and OpenOffice, use
            scheme 5.

    Case-insensitive named aliases are also supported: you can examine
    quartiles.aliases for a mapping of names to schemes.
    """
    d = sorted(data)
    return _quartiles(d, scheme)


# TODO make this a public function?
def _quartiles(data, scheme=None):
    """Return the quartiles of sorted data.

    data must be sorted list. See ``quartiles`` for further details.
    """
    n = len(data)
    if n < 3:
        raise _StatsError(
        'need at least 3 items to split data into quartiles')
    # Select a method.
    if scheme is None: scheme = QUARTILE_DEFAULT
    if isinstance(scheme, str):
        key = quartiles.aliases.get(scheme.lower())
    else:
        key = scheme
    func = _Quartiles.QUARTILE_MAP.get(key)
    if func is None:
        raise _StatsError('unrecognised scheme `%s`' % scheme)
    return func(data)

# TODO make this a read-only view of the dict?
quartiles.aliases = _quartiles.aliases = _Quartiles.QUARTILE_ALIASES


@sorted_data
def quantile(data, p, scheme=None):
    """quantile(data, p [, scheme]) -> value

    Return the value which is some fraction p of the way into data after
    sorting. data must be an iterable of numeric values, with at least two
    items. p must be a number between 0 and 1 inclusive. The result returned
    by quantile is the data point, or the interpolated data point, such that
    a fraction p of the data is less than that value.

    >>> data = [2.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    >>> quantile(data, 0.75)
    5.0


    Interpolation
    =============

    In general the quantile will not fall exactly on a data point. When that
    happens, the value returned is interpolated from the data points nearest
    the calculated position. There are a wide variety of interpolation methods
    used in the statistics literature, and quantile() allows you to choose
    between them using the optional argument scheme.

    >>> quantile(data, 0.75, scheme=4)
    4.5
    >>> quantile(data, 0.75, scheme=7)
    4.75

    scheme can be either an integer scheme number (see table below), a tuple
    of four numeric parameters, or a case-insensitive string alias for either
    of these. You can examine quantiles.aliases for a mapping of names to
    scheme numbers or parameters.

        WARNING: The use of arbitrary values as a four-parameter
        scheme is not recommended! Although quantile will calculate
        a result using them, the result is unlikely to be meaningful
        or statistically useful.

    Integer schemes 1-9 are equivalent to R's quantile types with the same
    number. These are also equivalent to Mathematica's parameterized quartile
    function with parameters shown:

    scheme  parameters   Description
    ======  ===========  ====================================================
    1       0,0,1,0      inverse of the empirical CDF
    2       n/a          inverse of empirical CDF with averaging
    3       1/2,0,0,0    closest actual observation
    4       0,0,0,1      linear interpolation of the empirical CDF
    5       1/2,0,0,1    Hazen's model (like Matlab's PRCTILE function)
    6       0,1,0,1      Weibull quantile
    7       1,-1,0,1     interpolation over range divided into n-1 intervals
    8       1/3,1/3,0,1  interpolation of the approximate medians
    9       3/8,1/4,0,1  approx. unbiased estimate for a normal distribution
    10      n/a          least expected square deviation relative to p

    Notes:

        (a) If scheme is missing or None, the default is taken from the
            global variable QUANTILE_DEFAULT (set to 1 by default).
        (b) Scheme 1 ensures that the values returned are always data points,
            and is the default used by Mathematica.
        (c) Scheme 5 is equivalent to Matlab's PRCTILE function.
        (d) Scheme 6 is equivalent to the method used by Minitab.
        (e) Scheme 7 is the default used by programming languages R and S,
            and is the method used by Microsoft Excel and OpenOffice.
        (f) Scheme 8 is recommended by Hyndman and Fan (1996).

    Example of using a scheme written in the parameterized form used by
    Mathematica:

    >>> data = [1, 2, 3, 3, 4, 5, 7, 9, 12, 12]
    >>> quantile(data, 0.2, scheme=(1, -1, 0, 1))  # First quintile.
    2.8

    This can also be written using an alias:

    >>> quantile(data, 0.2, scheme='excel')
    2.8

    """
    d = sorted(data)
    return _quantile(d, p, scheme)


# TODO make this a public function?
def _quantile(data, p, scheme=None):
    """Return the quantile from sorted data.

    data must be a sorted list. For additional details, see ``quantile``.
    """
    # More details here:
    # http://stat.ethz.ch/R-manual/R-devel/library/stats/html/quantile.html
    # http://en.wikipedia.org/wiki/Quantile
    if not 0.0 <= p <= 1.0:
        raise _StatsError(
        'quantile argument must be between 0.0 and 1.0')
    if len(data) < 2:
        raise _StatsError(
        'need at least 2 items to split data into quantiles')
    # Select a scheme.
    if scheme is None: scheme = QUANTILE_DEFAULT
    if isinstance(scheme, str):
        key = quantile.aliases.get(scheme.lower())
    else:
        key = scheme
    if isinstance(key, tuple) and len(key) == 4:
        return _parametrized_quantile(key, data, p)
    else:
        func = _Quantiles.QUANTILE_MAP.get(key)
        if func is None:
            raise _StatsError('unrecognised scheme `%s`' % scheme)
        return func(data, p)

# TODO make this a read-only view of the dict?
quantile.aliases = _quantile.aliases = _Quantiles.QUANTILE_ALIASES


def decile(data, d, scheme=None):
    """Return the dth decile of data, for integer d between 0 and 10.

    >>> data = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    >>> decile(data, 7, scheme=1)
    14

    See function quantile for details about the optional argument scheme.
    """
    stats.utils._validate_int(d)
    if not 0 <= d <= 10:
        raise ValueError('decile argument d must be between 0 and 10')
    from fractions import Fraction
    return quantile(data, Fraction(d, 10), scheme)


def percentile(data, p, scheme=None):
    """Return the pth percentile of data, for integer p between 0 and 100.

    >>> import builtins; data = builtins.range(1, 201)
    >>> percentile(data, 7, scheme=1)
    14

    See function quantile for details about the optional argument scheme.
    """
    stats.utils._validate_int(p)
    if not 0 <= p <= 100:
        raise ValueError('percentile argument p must be between 0 and 100')
    from fractions import Fraction
    return quantile(data, Fraction(p, 100), scheme)

