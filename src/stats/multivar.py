#!/usr/bin/env python3

##  Copyright (c) 2011 Steven D'Aprano.
##  See the file __init__.py for the licence terms for this software.

"""
Multivariate statistics.

"""

__all__ = [
    'qcorr', 'corr', 'pcov', 'cov', 'errsumsq', 'linr',
    ]

import collections
import functools
import itertools
import math

import stats

from stats import StatsError
from stats.utils import as_sequence, add_partial, _generalised_sum, \
    _sum_sq_deviations, _sum_prod_deviations

# === Helper and utility functions ===

class _Multivariate:
    # Helpers for dealing with multivariate functions.

    def __new__(cls):
        raise RuntimeError('namespace, do not instantiate')

    def split(xdata, ydata=None):
        """Helper function which splits xydata into (xdata, ydata)."""
        # The two-argument case is easy -- just pass them unchanged.
        if ydata is not None:
            xdata = as_sequence(xdata)
            ydata = as_sequence(ydata)
            if len(xdata) < len(ydata):
                ydata = ydata[:len(xdata)]
            elif len(xdata) > len(ydata):
                xdata = xdata[:len(ydata)]
            assert len(xdata) == len(ydata)
            return (xdata, ydata)
        # The single argument case could be either [x0, x1, x2, ...] or
        # [(x0, y0), (x1, y1), (x2, y2), ...]. We decide which it is by
        # looking at the first item, and treating it as canonical.
        it = iter(xdata)
        try:
            first = next(it)
        except StopIteration:
            # If the iterable is empty, return two empty lists.
            return ([], [])
        # If we get here, we know we have a single iterable argument with at
        # least one item. Does it look like a sequence of (x,y) values, or
        # like a sequence of x values?
        try:
            n = len(first)
        except TypeError:
            # Looks like we're dealing with the case [x0, x1, x2, ...]
            # This isn't exactly *multivariate*, but we support it anyway.
            # We leave it up to the caller to decide what to do with the
            # fake y values.
            xdata = [first]
            xdata.extend(it)
            return (xdata, [None]*len(xdata))
        # Looks like [(x0, y0), (x1, y1), (x2, y2), ...]
        # Here we expect that each point has two items, and fail if not.
        if n != 2:
            raise TypeError('expecting 2-tuple (x, y) but got %d-tuple' % n)
        xlist = [first[0]]
        ylist = [first[1]]
        for x,y in it:
            xlist.append(x)
            ylist.append(y)
        assert len(xlist) == len(ylist)
        return (xlist, ylist)

    def merge(xdata, ydata=None):
        """Helper function which merges xdata, ydata into xydata."""
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
        # least one item. Does it look like a sequence of (x,y) values, or
        # like a sequence of x values?
        try:
            len(first)
        except TypeError:
            # Looks like we're dealing with the case [x0, x1, x2, ...]
            first = (first, None)
            tail = ((x, None) for x in it)
            return itertools.chain([first], tail)
        # Looks like [(x0, y0), (x1, y1), (x2, y2), ...]
        # Notice that we DON'T care how many items are in the data points
        # here, we postpone dealing with any mismatches to later.
        return itertools.chain([first], it)

    def split_xydata(func):
        """Decorator to split a single (x,y) data iterable into separate x
        and y iterables.
        """
        @functools.wraps(func)
        def inner(xdata, ydata=None):
            xdata, ydata = _Multivariate.split(xdata, ydata)
            return func(xdata, ydata)
        return inner

    def merge_xydata(func):
        """Decorator to merge separate x, y data iterables into a single
        (x,y) iterator.
        """
        @functools.wraps(func)
        def inner(xdata, ydata=None):
            xydata = _Multivariate.merge(xdata, ydata)
            return func(xydata)
        return inner



# === Simple multivariate statistics ===

@_Multivariate.split_xydata
def qcorr(xdata, ydata):
    """Return the Q correlation coefficient of (x, y) data.

    If ydata is None or not given, then xdata must be an iterable of (x, y)
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
    relationship), and some value between -1 and +1 in nearly all other cases,
    indicating the degree of linear dependence between the variables:

    >>> qcorr([(1, 1), (2, 3), (2, 1), (3, 5), (4, 2), (5, 3), (6, 4)])
    0.5

    In the case where all points are on the median lines, returns a float NAN.
    """
    from stats.order import median
    n = len(xdata)
    assert n == len(ydata)
    if n == 0:
        raise StatsError('Q correlation requires non-empty data')
    xmed = median(xdata)
    ymed = median(ydata)
    # Traditionally, we count the values in each quadrant, but in fact we
    # really only need to count the diagonals: quadrants 1 and 3 together,
    # and quadrants 2 and 4 together.
    quad13 = quad24 = skipped = 0
    for x,y in zip(xdata, ydata):
        if x > xmed:
            if y > ymed:  quad13 += 1
            elif y < ymed:  quad24 += 1
            else:  skipped += 1
        elif x < xmed:
            if y > ymed:  quad24 += 1
            elif y < ymed:  quad13 += 1
            else:  skipped += 1
        else:  skipped += 1
    assert quad13 + quad24 + skipped == n
    if skipped == n:
        return float('nan')
    q = (quad13 - quad24)/(n - skipped)
    assert -1.0 <= q <= 1.0
    return q


@_Multivariate.split_xydata
def corr(xdata, ydata):
    """corr(xydata) -> float
    corr(xdata, ydata) -> float

    Return the sample Pearson's Correlation Coefficient of (x,y) data.

    If ydata is None or not given, then xdata must be an iterable of (x, y)
    pairs. Otherwise, both xdata and ydata must be iterables of values, which
    will be truncated to the shorter of the two.

    >>> corr([(0.1, 2.3), (0.5, 2.7), (1.2, 3.1), (1.7, 2.9)])
    ... #doctest: +ELLIPSIS
    0.827429009335...

    The Pearson correlation is +1 in the case of a perfect positive
    correlation (i.e. an increasing linear relationship), -1 in the case of
    a perfect anti-correlation (i.e. a decreasing linear relationship), and
    some value between -1 and 1 in all other cases, indicating the degree
    of linear dependence between the variables.

    >>> xdata = [1, 2, 3, 4, 5, 6]
    >>> ydata = [2*x for x in xdata]  # Perfect correlation.
    >>> corr(xdata, ydata)
    1.0
    >>> corr(xdata, [5-y for y in ydata])  # Perfect anti-correlation.
    -1.0

    If there are not at least two data points, or if either all the x values
    or all the y values are equal, StatsError is raised.
    """
    n = len(xdata)
    assert n == len(ydata)
    if n < 2:
        raise StatsError(
            'correlation requires at least two data points, got %d' % n)
    # First pass is to determine the means.
    mx = stats.mean(xdata)
    my = stats.mean(ydata)
    # Second pass to determine the standard deviations.
    sx = stats.stdev(xdata, mx)
    sy = stats.stdev(ydata, my)
    if sx == 0:
        raise StatsError('all x values are equal')
    if sy == 0:
        raise StatsError('all y values are equal')
    # Third pass to calculate the correlation coefficient.
    ap = add_partial
    total = []
    for x, y in zip(xdata, ydata):
        term = ((x-mx)/sx) * ((y-my)/sy)
        ap(term, total)
    r = math.fsum(total)/(n-1)
    assert -1 <= r <= r
    return r


# Alternate implementation.
def _corr2(xdata, ydata=None):
    raise NotImplementedError('do not use this')
    #t = xysums(xdata, ydata)
    #r = t.Sxy/math.sqrt(t.Sxx*t.Syy)


@_Multivariate.split_xydata
def pcov(xdata, ydata=None):
    """Return the population covariance between (x, y) data.

    >>> pcov([0.75, 1.5, 2.5, 2.75, 2.75], [0.25, 1.1, 2.8, 2.95, 3.25])
    ... #doctest: +ELLIPSIS
    0.93399999999...
    >>> pcov([(0.1, 2.3), (0.5, 2.7), (1.2, 3.1), (1.7, 2.9)])
    0.15125

    """
    n, s = _sum_prod_deviations(zip(xdata, ydata), None, None)
    if n > 0:
        return s/n
    else:
        raise StatsError('population covariance requires at least one point')
    #t = xysums(xdata, ydata)
    #return t.Sxy/(t.n**2)


@_Multivariate.split_xydata
def cov(xdata, ydata):
    """Return the sample covariance between (x, y) data.

    >>> cov([(0.1, 2.3), (0.5, 2.7), (1.2, 3.1), (1.7, 2.9)])
    ... #doctest: +ELLIPSIS
    0.201666666666...

    >>> cov([0.75, 1.5, 2.5, 2.75, 2.75], [0.25, 1.1, 2.8, 2.95, 3.25])
    ... #doctest: +ELLIPSIS
    1.1675
    >>> cov([(0.1, 2.3), (0.5, 2.7), (1.2, 3.1), (1.7, 2.9)])
    ... #doctest: +ELLIPSIS
    0.201666666666...

    Covariance reduces down to standard variance when applied to the same
    data as both the x and y values:

    >>> data = [1.2, 0.75, 1.5, 2.45, 1.75]
    >>> cov(data, data)  #doctest: +ELLIPSIS
    0.40325000000...
    >>> stats.variance(data)  #doctest: +ELLIPSIS
    0.40325000000...

    """
    n, s = _sum_prod_deviations(zip(xdata, ydata), None, None)
    if n > 1:
        return s/(n-1)
    else:
        raise StatsError('sample covariance requires at least two points')
    # t = xysums(xdata, ydata)
    # return t.Sxy/(t.n*(t.n-1))


def _SP(xdata, mx, ydata, my):
    """SP = sum of product of deviations.
    Helper function for calculating covariance directly.
    """
    if mx is None:
        # Two pass algorithm.
        xdata = as_sequence(xdata)
        mx = stats.mean(xdata)
    if my is None:
        # Two pass algorithm.
        ydata = as_sequence(ydata)
        my = stats.mean(ydata)
    return _generalised_sum(zip(xdata, ydata), lambda t: (t[0]-mx)*(t[1]-my))


@_Multivariate.split_xydata
def errsumsq(xdata, ydata):
    """Return the error sum of squares of (x,y) data.

    The error sum of squares, or residual sum of squares, is the estimated
    variance of the least-squares linear regression line of (x,y) data.

    >>> errsumsq([1, 2, 3, 4], [1.5, 1.5, 3.5, 3.5])
    0.4

    """
    t = xysums(xdata, ydata)
    return (t.Sxx*t.Syy - (t.Sxy**2))/(t.n*(t.n-2)*t.Sxx)


@_Multivariate.split_xydata
def linr(xdata, ydata):
    """Return the linear regression coefficients a and b for (x,y) data.

    Returns the y-intercept and slope of the straight line of the least-
    squared regression line, that is, the line which minimises the sum of
    the squares of the errors between the actual and calculated y values.

    >>> xdata = [0.0, 0.25, 1.25, 1.75, 2.5, 2.75]
    >>> ydata = [1.5*x + 0.25 for x in xdata]
    >>> linr(xdata, ydata)
    (0.25, 1.5)

    """
    t = xysums(xdata, ydata)
    if t.n < 2:
        raise StatsError('regression line requires at least two points')
    b = t.Sxy/t.Sxx
    a = t.sumy/t.n - b*t.sumx/t.n
    return (a, b)


# === Sums and products ===

@_Multivariate.merge_xydata
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
    n, ss = _sum_sq_deviations((x for (x, y) in xydata), None)
    return ss*n


@_Multivariate.merge_xydata
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
    # set to None. (See the merge_xydata function.) We have to detect
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
    n, ss = _sum_sq_deviations((y for (x, y) in xydata), None)
    return ss*n


@_Multivariate.merge_xydata
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

