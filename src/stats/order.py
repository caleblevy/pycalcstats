#!/usr/bin/env python3

##  Copyright (c) 2011 Steven D'Aprano.
##  See __init__.py for the licence terms for this software.

"""
=======================
Order statistics module
=======================

The ``stats.order`` module calculates order statistics and related functions.

The kth order statistic of sample data is the kth smallest value in the
sample. For example, you might ask "what is the highest income of the
poorest 10% of workers?" -- that would be equivalent to asking for the 1st
decile (or the 10th percentile, or just the 0.1 quantile).

This module provides the following order statistics and related functions:

    Function            Description
    ==================  ===============================================
    decile              The specified 10-fractile of the data.
    fivenum             Tukey's five number summary.
    iqr                 Inter-Quartile Range.
    mad                 Median Average Deviation.
    median              The middle of the data.
    midhinge            The midpoint between the hinges.
    midrange            The midpoint of the smallest and largest values.
    minmax              Minimum and maximum of the arguments.
    percentile          The specified 100-fractile of the data.
    quantile            An arbitrary quantile.
    quartile_skewness   Skewness of the data calculated from quartiles.
    quartiles           The 4-fractiles of the data.
    range               The largest value minus the smallest value.
    trimean             Tukey's trimean.

The ``minmax`` function is an alias to the function of the same name in
the ``stats`` module.


"""
# TODO: investigate finding median and other fractiles without sorting,
# e.g. QuickSelect, ranking, etc. See:
# http://mail.gnome.org/archives/gnumeric-list/2007-February/msg00023.html
# http://mail.gnome.org/archives/gnumeric-list/2007-February/msg00041.html
#
# Add support for fractiles (and median?) withour sorting each time.
# Add Raymond Hettinger's running median recipe?


__all__ = [
    'decile', 'fivenum', 'iqr', 'mad', 'median', 'midhinge', 'midrange',
    'minmax', 'percentile', 'quantile', 'quartile_skewness', 'quartiles',
    'range', 'trimean',
    ]


import collections
import functools
import math
import types

import stats

from stats import minmax


# === Global variables ===

# We need this to work around doctest cleverness, since otherwise it won't
# look inside namespace objects for tests.
__test__ = {}


# === Private utilities ===

def _namespace(obj):
    """Decorator to build a namespace."""
    class Namespace(types.ModuleType):
        def __repr__(self):
            return "<namespace '%s'>" % self.__name__

    ns = Namespace(obj.__name__)
    ns.__doc__ = obj.__doc__
    for name in dir(obj):
        if not name.startswith('__'):
            attr = getattr(obj, name)
            if type(attr) is types.MethodType:
                attr = attr.__func__
            setattr(ns, name, attr)
    __test__[ns.__name__] = ns
    return ns


def _interpolate(data, x):
    """Return the interpolated value of data at possibly fractional
    non-negative index x.

    If x is an integer value, returns data[x]. If x is a non-integer, the
    value returned is estimated between data[x] and data[x+1], using linear
    interpolation.

    >>> _interpolate([1, 3, 5, 7, 9], 3)
    7
    >>> _interpolate([1, 3, 5, 7, 9], 3.75)
    8.5

    Only non-negative indices are supported. Behaviour for negative x is
    unspecified.
    """
    assert x >= 0
    i, f = int(x), x%1
    if f:
        a, b = data[i], data[i+1]
        return a + f*(b-a)
    else:
        return data[i]


def _round_halfeven(x):
    """Round non-negative float x to the nearest integer value, with ties
    rounded to an even value. Also known as Banker's rounding.

    >>> [_round_halfeven(x) for x in (2.0, 2.1, 2.5, 2.9, 3.5)]
    [2, 2, 2, 3, 4]

    Only non-negative x values are supported. Behaviour for negative x is
    unspecified.
    """
    assert x >= 0
    n, f = int(x), x%1
    if f > 0.5:
        return n+1
    elif f < 0.5:
        return n
    else:
        return n+1 if n%2 else n


def _round_halfup(x):
    """Round non-negative float x to the nearest integer value, with ties
    rounded up.

    >>> [_round_halfup(x) for x in (2.0, 2.1, 2.5, 2.9, 3.5)]
    [2, 2, 3, 3, 4]

    Only non-negative x values are supported. Behaviour for negative x is
    unspecified.
    """
    assert x >= 0
    return math.floor(x + 0.5)


def _inject_aliases(ns):
    """Decorator that adds a reference to ns.ALIASES_MAP to the decorated
    function.

    >>> c = type('C', (object,), {})()
    >>> c.ALIASES_MAP = {}
    >>> @_inject_aliases(c)
    ... def f():
    ...     pass
    ...
    >>> f.aliases is c.ALIASES_MAP
    True

    """
    def decorator(func):
        func.aliases = ns.ALIASES_MAP
        return func
    return decorator


def _get_scheme_func(ns, scheme):
    """Return a function specified by the given scheme."""
    if isinstance(scheme, str):
        scheme = ns.ALIASES_MAP[scheme.lower()]
    return ns.FUNC_MAP[scheme]


# === Order statistics ===


# Fractiles: medians, quartiles, quantiles and hinges
# ---------------------------------------------------
#
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
# Quantiles and percentiles also have a plethora of calculation methods.
# R includes nine different methods for quantiles. Mathematica uses a
# parameterized quantile function capable of matching eight of those nine
# methods. Wikipedia lists a tenth method. The Haskell statistics package
# includes six. There are probably others I don't know of. And then there
# are grouped and weighted data, all of which have their own methods too :(
#
# The most frustrating part of this is that most examples of fractile use
# don't give any hint as to which calculation method is used, or even that
# there is a choice to be made.
#
# The approach used by R is to specify the calculation "type" as an optional
# parameter of the quantile function: quantile(x, probs ... type = 7, ...).
# SAS uses a similar approach, using different numbers. I take a similar
# approach. Notable differences are:
#
#   * I use the term "scheme" rather than "type" or "method";
#   * I support user-extensible string aliases (names) as well as
#     numeric codes;
#   * Mathematica-style parameterized quantiles are also supported.
#   * Each of the ``median``, ``quartiles`` and ``quantile`` functions
#     have their own set of schemes. See each function's docstring for
#     details.


# -- Private fractile functions --

@_namespace
class _Median:
    """Private namespace for median calculation methods. All functions and
    attributes in this namespace are private and subject to change without
    notice.

    All functions assume that their data argument is a sorted list. If that
    assumption is violated, behaviour is unspecified.
    """

    def standard_median(data):
        """Return median using the standard mean-of-middle-two method.

        >>> _Median.standard_median([1, 3, 5])
        3
        >>> _Median.standard_median([1, 3, 5, 7])
        4.0

        """
        n = len(data)
        if n%2 == 1:
            return data[n//2]
        else:
            m = n//2
            return (data[m - 1] + data[m])/2

    def low_median(data):
        """Return the low median.

        >>> _Median.low_median([1, 3, 5])
        3
        >>> _Median.low_median([1, 3, 5, 7])
        3

        """
        n = len(data)
        if n%2 == 1:
            return data[n//2]
        else:
            return data[n//2 - 1]

    def high_median(data):
        """Return the high median.

        >>> _Median.high_median([1, 3, 5])
        3
        >>> _Median.high_median([1, 3, 5, 7])
        5

        """
        n = len(data)
        return data[n//2]

    def dup_median(data):
        """"Median with adjustment for duplicate values.

        >>> _Median.dup_median([2, 2, 3, 4])
        2.3333333333333335
        >>> _Median.dup_median([1, 3, 3, 5, 7])
        3
        >>> _Median.dup_median([1, 3, 3, 5, 7, 9])
        3.6666666666666665
        >>> _Median.dup_median([1, 3, 3, 3, 5, 5, 7, 9])
        3.8

        """
        n = len(data)
        m = n//2
        if n%2 == 1:
            return data[m]
        a, b = data[m-1], data[m]
        if a == b:
            return a
        ca = data.count(a)
        cb = data.count(b)
        return (ca*a + cb*b)/(ca + cb)

    # Set up mappings for schemes and aliases. Canonical schemes are the
    # numeric codes, mapped to a function in FUNC_MAP. Aliases are given as
    # lowercase strings in ALIASES_MAP.
    FUNC_MAP = {
        1: standard_median,
        2: low_median,
        3: high_median,
        4: dup_median,
        }
    ALIASES_MAP = {
        'dup': 4,
        'high': 3,
        'low': 2,
        'standard': 1,
        'std': 1,
        }
    assert all(alias==alias.lower() for alias in ALIASES_MAP)


@_namespace
class _Quartiles:
    """Private namespace for median calculation methods. All functions and
    attributes in this namespace are private and subject to change without
    notice.

    All functions assume that their data argument is a sorted list. If that
    assumption is violated, behaviour is unspecified.
    """

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
        n = len(data)
        M = _round_halfeven((n+1)/2)
        L = _round_halfup((n+1)/4)
        U = n+1-L
        assert U == math.ceil(3*(n+1)/4 - 0.5)  # Round half down.
        return (data[L-1], data[M-1], data[U-1])

    def minitab(data):
        """Return sample quartiles using the method used by Minitab."""
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

    # Set up mappings for schemes and aliases. Canonical schemes are the
    # numeric codes, mapped to a function in FUNC_MAP. Aliases are given as
    # lowercase strings in ALIASES_MAP.
    FUNC_MAP = {
        1: inclusive,
        2: exclusive,
        3: ms,
        4: minitab,
        5: excel,
        6: langford,
        }
    ALIASES_MAP = {
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
    assert all(alias==alias.lower() for alias in ALIASES_MAP)


@_namespace
class _Quantile:
    """Private namespace for quantile calculation methods. All functions and
    attributes in this namespace are private and subject to change without
    notice.

    All functions assume that their data argument is a sorted list, and that
    the p argument is a fraction 0 <= p <= 1. If either assumption is
    violated, behaviour is unspecified.
    """

    # The functions r1...r9 implement R's quantile types 1...9 respectively.
    # Except for r2, they are also equivalent to Mathematica's parametrized
    # quantile function: http://mathworld.wolfram.com/Quantile.html

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
        if i == j:
            return data[i-1]
        return (data[i-1] + data[j-1])/2

    def r3(data, p):  # FIXME
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

    class D(dict):
        @staticmethod
        def param_factory(parameters):
            """Factory function returning a parameterized quantile function
            similar to the Mathematica Quantile() function.

            param_factory((a,b,c,d)) -> function
            function(data, p) -> value

            The returned function takes two arguments, a sorted list ``data``
            and a fraction ``p`` between 0 and 1 inclusive, and returns the
            pth quantile of data based on the parameters supplied to the
            factory.

            >>> from builtins import range
            >>> data = list(range(1, 21))
            >>> func = _Quantile.D.param_factory((0, 0, 1, 0))
            >>> func(data, 0.3)
            6.0
            >>> func = _Quantile.D.param_factory((1/2, 0, 0, 1))
            >>> func(data, 0.3)
            6.5

            WARNING: While this function will accept arbitrary numberic
            values for the parameters, not all such combinations are
            meaningful:

            >>> bad_func = _Quantile.D.param_factory((1, 1, 1, 1))
            >>> bad_func([1, 2], 0.3)
            2.9

            """
            assert isinstance(parameters, tuple)
            a, b, c, d = parameters
            def func(data, p):
                """Parameterized quantile function."""
                # More details here:
                # http://reference.wolfram.com/mathematica/ref/Quantile.html
                # http://mathworld.wolfram.com/Quantile.html
                assert isinstance(data, list)
                assert 0 <= p <= 1
                n = len(data)
                h = a + (n+b)*p
                i = max(1, min(math.floor(h), n))
                j = max(1, min(math.ceil(h), n))
                x = data[i-1]
                y = data[j-1]
                return x + (y - x)*(c + d*(h%1))
            return func

        def __missing__(self, key):
            if isinstance(key, tuple) and len(key) == 4:
                return self.param_factory(key)
            raise KeyError('bad scheme')

    # Set up mappings for schemes and aliases. Canonical schemes are the
    # numeric codes, mapped to a function in FUNC_MAP. Aliases are given as
    # lowercase strings in ALIASES_MAP.
    FUNC_MAP = D({
        # Schemes 1-9 must match the R calculation type with the same
        # number. In other words, don't touch keys 1-9!
        1:r1,  2:r2,  3:r3,  4:r4,  5:r5,  6:r6,  7:r7,  8:r8,  9:r9,
        10: qlsd,
        })
    ALIASES_MAP = {
        'cdf': 2,
        'excel': 7,  # Method used by Excel, OpenOffice and Gnumeric.
        'h&f': 8,
        'hyndman': 8,
        'matlab': 5,  # As used by Matlab.
        'minitab': 6,  # As used by Minitab.
        's': 7,  # As used by S.
        'sas-1': 4,  # Methods available in SAS.
        'sas-2': 3,
        'sas-3': 1,
        'sas-4': 6,
        'sas-5': 2,
        'spss': 6, # As used by SPSS.
        }
    assert all(alias==alias.lower() for alias in ALIASES_MAP)

# -- Public fractile functions --

@_inject_aliases(_Median)
def median(data, scheme=1):
    """Returns the median (middle) value of an iterable of numbers.

    >>> median([3.0, 5.0, 2.0])
    3.0

    The median is the middle data point in a sorted sequence of values, and
    is commonly used as an average or estimate of central tendency. It is
    more robust than the mean for data that contains outliers -- if your
    data contains a few items that are extremely small, or extremely large,
    compared to the rest of the data, the median will be more representative
    of the data than the mean.

    There are a number of alternative methods for calculating the median.
    The optional argument ``scheme`` allows you to specify which calculation
    method to use. In general, the median is always the middle value if
    there are an odd number of points. For even number of points, the middle
    falls between two values, and some form of interpolation is needed. The
    supported schemes are:

    scheme  Description
    ======  =================================================================
    1       The mean of the two values straddling the middle (the default).
    2       Low median: the element just below the middle.
    3       High median: the element just above the middle.
    4       Mean of central values with an adjustment for duplicates.

    Case-insensitive named aliases are also supported: you can examine
    median.aliases for a mapping of names to schemes.
    """
    func = _get_scheme_func(_Median, scheme)
    if isinstance(data, str):
        raise TypeError('data argument cannot be a string')
    data = sorted(data)
    if len(data) == 0:
        raise stats.StatsError('no median for empty iterable')
    else:
        return func(data)


@_inject_aliases(_Quartiles)
def quartiles(data, scheme=1):
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

        (a) Scheme 1 (the default) is equivalent to Tukey's hinges
            (H1, M, H2).
        (b) Scheme 2 is used by Texas Instruments calculators starting with
            model TI-85.
        (c) Scheme 3 ensures that the values returned are always data points.
        (d) Schemes 4 and 5 use linear interpolation between items.
        (e) For compatibility with Microsoft Excel and OpenOffice, use
            scheme 5.

    Case-insensitive named aliases are also supported: you can examine
    quartiles.aliases for a mapping of names to schemes.
    """
    func = _get_scheme_func(_Quartiles, scheme)
    if isinstance(data, str):
        raise TypeError('data argument cannot be a string')
    data = sorted(data)
    if len(data) < 3:
        raise stats.StatsError(
        'need at least 3 items to split data into quartiles')
    return func(data)


@_inject_aliases(_Median)
def quantile(data, p, scheme=1):
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
    the calculated position. There are a wide variety of interpolation
    methods used in the statistics literature, and quantile() allows you to
    choose between them using the optional argument scheme.

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
    1       0,0,1,0      inverse of the empirical CDF (the default)
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

        (a) Scheme 1 ensures that the values returned are always data points,
            and is the default used by Mathematica.
        (b) Scheme 5 is equivalent to Matlab's PRCTILE function.
        (c) Scheme 6 is equivalent to the method used by Minitab.
        (d) Scheme 7 is the default used by programming languages R and S,
            and is the method used by Microsoft Excel and OpenOffice.
        (e) Scheme 8 is recommended by Hyndman and Fan (1996).

    Example of using a scheme written in the parameterized form used by
    Mathematica:

    >>> data = [1, 2, 3, 3, 4, 5, 7, 9, 12, 12]
    >>> quantile(data, 0.2, scheme=(1, -1, 0, 1))  # Get the first quintile.
    2.8

    This can also be written using an alias:

    >>> quantile(data, 0.2, scheme='excel')
    2.8

    """
    # More details here:
    # http://stat.ethz.ch/R-manual/R-devel/library/stats/html/quantile.html
    # http://en.wikipedia.org/wiki/Quantile
    func = _get_scheme_func(_Quantile, scheme)
    if isinstance(data, str):
        raise TypeError('data argument cannot be a string')
    if not 0.0 <= p <= 1.0:
        raise stats.StatsError(
        'quantile argument must be between 0.0 and 1.0')
    data = sorted(data)
    if len(data) < 2:
        raise stats.StatsError(
        'need at least 2 items to split data into quantiles')
    return func(data, p)


# -- Convenience functions for fractiles --

def fivenum(data):
    """Return Tukey's five number summary from data.

    The five summary numbers are:

        minimum, lower-hinge, median, upper-hinge, maximum


    >>> tuple(fivenum([2, 4, 6, 8, 10, 12, 14, 16, 18]))
    (2, 6, 10, 14, 18)

    The summary is a namedtuple with the following fields:

        minimum
        lower_hinge
        median
        upper_hinge
        maximum

    If the data has length N of the form ``4n+5`` (e.g. 5, 9, 13, 17...)
    then the hinges can be visualised by writing out the sorted data in the
    shape of a W, where each limb of the W is equal is length. For example,
    the data (A,B,C,...,M) has N=13 and would be written out like this:

        A           G           M
          B       F   H       L
            C   E       I   K
              D           J

    The hinges are D, G and J and the fivenum summary is (A, D, G, J, M).

    For data with length that doesn't match ``4n+5``, the three hinges are
    interpolated. They are equivalent to ``quartiles`` called with scheme=1.
    """
    if isinstance(data, str):
        raise TypeError('data argument cannot be a string')
    data = sorted(data)
    a, b = minmax(data)
    h1, m, h2 = quartiles(data, scheme=1)
    summary = collections.namedtuple('fivenum',
                'minimum lower_hinge median upper_hinge maximum')
    return summary(a, h1, m, h2, b)


def decile(data, d, scheme=1):
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


def percentile(data, p, scheme=1):
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



# Other measures of central tendency
# ----------------------------------

def midrange(data):
    """Returns the midrange of a sequence of numbers.

    >>> midrange([2.0, 3.0, 3.5, 4.5, 7.5])
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
    H1, _, H2 = quartiles(data, scheme=1)
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
    H1, M, H2 = quartiles(data, scheme=1)
    return (H1 + 2*M + H2)/4


# Measures of spread (dispersion or variability)
# ----------------------------------------------

def range(data, interval=0):
    """range(iterable [, interval=0]) -> sample range R of data

    The range R is the difference between the smallest and largest element
    in the given sample. It is an unbiased but weak measure of variability,
    and is frequently used in process control applications.

    >>> range([1.0, 3.5, 7.5, 2.0, 0.25])
    7.25

    For N > 15, the sampling distribution of R becomes unstable and it is
    wise to treat the sample range with caution.

    An even better measure of variability is R/d2, where d2 is a value
    that depends only on N. For samples taken from a normally-distributed
    population, the d2 values are available by looking up N in the dict
    ``range.d2``. For small N (say, up to about 10) R/d2 makes a good
    estimator of the population standard deviation.


    Correction for binned or rounded data
    -------------------------------------

    If the data points have been uniformly rounded (perhaps by binning, or
    by rounding to a fixed number of decimal places, or simply due to
    measurement error), the samples represent intervals rather than exact
    values. E.g. if x=1.2 is given to one decimal place, x could actually
    be any number between 1.15 and 1.25. In this case, it is appropriate to
    make an adjustment to the sample range by taking into account the width
    of the data interval:

    >>> range([1.2, 3.0, 1.5, 2.4, 0.2], 0.1)
    2.9

    The ``interval`` argument is optional, with default value of 0. If
    given, it must be a non-negative number.

    No attempt is made to check that the data points actually are consistent
    with the given interval.
    """
    if interval < 0:
        raise ValueError('interval must be non-negative')
    try:
        a, b = minmax(data)
    except ValueError as e:
        e.args = ('no range defined for empty iterables',)
        raise
    return b - a + interval

range.d2 = {
    2: 1.128,   3: 1.693,   4: 2.059,   5: 2.326,   6: 2.534,   7: 2.704,
    8: 2.847,   9: 2.970,   10: 3.078,  11: 3.173,  12: 3.258,  13: 3.336,
    14: 3.407,  15: 3.472,
    # Source: "Probability and Statistics for Engineers", Irwin Miller
    # and John E. Freund, Prentice-Hall, third edition.
    # See also http://www.itl.nist.gov/div898/handbook/pmc/section3/pmc321.htm
    }


def iqr(data, scheme=1):
    """Returns the Inter-Quartile Range of a sequence of numbers.

    >>> iqr([0.5, 2.25, 3.0, 4.5, 5.5, 6.5])
    3.25

    The IQR is the difference between the first and third quartile. The
    optional argument scheme is used to select the algorithm for calculating
    the quartiles. See the quartile function for further details.

    The IQR with scheme 1 (the default) is equivalent to Tukey's H-spread.
    """
    q1, _, q3 = quartiles(data, scheme)
    return q3 - q1


def mad(data, m=None, scheme=1, scale=1):
    """mad(iterable [, m=None [, scheme=1 [, scale=1]]]) -> value

    Returns the median absolute deviation (MAD) of data.

    >>> mad([1, 1, 2, 2, 4, 6, 9])
    1

    The MAD is the median of the absolute deviations from the median. For
    many sets of data, the MAD is close to half of the much simpler IQR,
    however MAD is a more robust measurement of spread than either IQR or
    standard deviation, and is less affected by outliers. MAD is also
    defined for distributions such as the Cauchy distribution which don't
    have either a mean or standard deviation.

    Arguments are:

    data    Iterable of data values.
    m       Optional centre location, nominally the median. If m is not
            given, or is None, the median is calculated from data.
    scheme  Select a calculation method for the median. See the ``median``
            function for further details.
    scale   Optional scale factor, by default no scale factor is applied.

    Common values used for ``scheme`` are 1 (the default) to use the standard
    median definition, 2 to use the "low median", or 3 to use the "high
    median".

    >>> data = [1, 1, 2, 4, 6, 9]
    >>> mad(data, scheme=1)
    2.0
    >>> mad(data, scheme=2)
    1
    >>> mad(data, scheme=3)
    3

    The MAD can also be used as a robust estimate for the standard deviation
    by multipying it by a scale factor. The scale factor can be passed
    directly as a numeric value, which is assumed to be positive but no check
    is applied. Other values accepted are:

    'normal'    Apply a scale factor of 1.4826, applicable to data from a
                normally distributed population.
    'uniform'   Apply a scale factor of approximately 1.1547, applicable
                to data from a uniform distribution.
    None, 'none' or not supplied:
                No scale factor is applied (the default).

    See ``mad.scaling`` for a mapping between scale factor names and values.
    """
    # Check for an appropriate scale factor.
    if isinstance(scale, str):
        f = mad.scaling.get(scale.lower())
        if f is None:
            raise stats.StatsError('unrecognised scale factor `%s`' % scale)
        scale = f
    elif scale is None:
        scale = 1
    if m is None:
        if not isinstance(data, (list, tuple)):
            data = list(data)
        m = median(data, scheme)
    med = median((abs(x - m) for x in data), scheme)
    return scale*med

mad.scaling = {
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

def quartile_skewness(q1, q2, q3):
    """Return the quartile skewness coefficient, or Bowley skewness, from
    the three quartiles q1, q2, q3.

    >>> quartile_skewness(1, 2, 5)
    0.5
    >>> quartile_skewness(1, 4, 5)
    -0.5

    """
    if not q1 <= q2 <= q3:
        raise stats.StatsError('quartiles must be ordered q1 <= q2 <= q3')
    if q1 == q2 == q3:
        return float('nan')
    skew = (q3 + q1 - 2*q2)/(q3 - q1)
    assert -1.0 <= skew <= 1.0
    return skew

