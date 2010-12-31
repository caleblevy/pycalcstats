#!/usr/bin/env python3

##  Package stats.py
##
##  Copyright (c) 2011 Steven D'Aprano.
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
Basic calculator statistics.

'Scientific calculator' statistics for Python 3.

Features:

(1) Standard calculator statistics such as mean and standard deviation:

    >>> mean([-1.0, 2.5, 3.25, 5.75])
    2.625
    >>> stdev([2.5, 3.25, 5.5, 11.25, 11.75])  #doctest: +ELLIPSIS
    4.38961843444...

(2) Single-pass variations on common statistics for use on large iterators
    with little or no loss of precision:

    >>> data = iter([2.5, 3.25, 5.5, 11.25, 11.75])
    >>> stdev1(data)  #doctest: +ELLIPSIS
    4.38961843444...

(3) Order statistics such as the median and quartiles:

    >>> median([6, 1, 5, 4, 2, 3])
    3.5
    >>> quartiles([2, 4, 5, 3, 1, 6])
    (2, 3.5, 5)

(4) Over forty statistics, including such functions as trimean and standard
    error of the mean:

    >>> trimean([15, 18, 20, 29, 35])
    21.75
    >>> sterrmean(3.25, 100, 1000)  #doctest: +ELLIPSIS
    0.30847634861...

"""


# Package metadata.
__version__ = "0.2.0a"
__date__ = "2011-01-?????????????????????????????????????????????????"
__author__ = "Steven D'Aprano"
__author_email__ = "steve+python@pearwood.info"


__all__ = ['StatsError', 'sum', 'mean', 'pvariance', 'variance', 'pstdev', 'stdev']

import collections

from . import utils


# === Exceptions ===

class StatsError(ValueError):
    pass


# === Basic univariate statistics ===

def sum(data, start=0):
    """sum(iterable_of_numbers [, start]) -> sum of numbers

    Return a high-precision sum of the given numbers.

    When passed a single sequence or iterator of numbers, sum() adds the
    numbers and returns the total:

    >>> sum([2.25, 4.5, -0.5, 1.0])
    7.25

    If optional argument start is given, it is added to the total. If the
    iterable is empty, start (defaulting to 0) is returned.

    The numbers are added using high-precision arithmetic that can avoid
    some sources of round-off error. In comparison, the builtin sum can
    suffer from catastrophic cancellation, for example the builtin
    sum([1, 1e100, 1, -1e100]*10000) may return zero instead of the correct
    value of 20000. This sum returns the correct result:

    >>> sum([1, 1e100, 1, -1e100] * 10000)
    20000.0

    """
    _, total = utils._generalised_sum(data, None)
    return total + start


def mean(data):
    """mean(iterable_of_numbers) -> mean of numbers

    Return the sample arithmetic mean of the given numbers.

    The arithmetic mean is the sum of the data divided by the number of data.
    It is commonly called "the average", although it is actually only one of
    many different averages. It is a measure of the central location of the
    data.

    When passed a single sequence or iterator of numbers, mean() adds the
    data points and returns the total divided by the number of data points:

    >>> mean([1.0, 2.0, 3.0, 4.0])
    2.5

    The sample mean is an unbiased estimator of the true population mean.
    However, the mean is strongly effected by outliers and is not a robust
    estimator for central location.
    """
    count, total = utils._generalised_sum(data, None)
    if count:
        return total/count
    else:
        raise StatsError('mean of empty sequence is not defined')


def pvariance(data, m=None):
    """pvariance(data [, m]) -> population variance of data.

    >>> pvariance([0.25, 0.5, 1.25, 1.25,
    ...           1.75, 2.75, 3.5])  #doctest: +ELLIPSIS
    1.17602040816...

    If you know the population mean, or an estimate of it, then you can pass
    the mean as the optional argument m. See also pstdev.

    The variance is a measure of the variability (spread or dispersion) of
    data. The population variance applies when data represents the entire
    relevant population. If it represents a statistical sample rather than
    the entire population, you should use variance instead.
    """
    n, ss = _SS(data, m)
    if n < 1:
        raise StatsError('population variance or standard deviation'
        ' requires at least one data point')
    return ss/n


def variance(data, m=None):
    """variance(data [, m]) -> sample variance of data.

    >>> variance([0.25, 0.5, 1.25, 1.25,
    ...           1.75, 2.75, 3.5])  #doctest: +ELLIPSIS
    1.37202380952...

    If you know the population mean, or an estimate of it, then you can pass
    the mean as the optional argument m. See also stdev.

    The variance is a measure of the variability (spread or dispersion) of
    data. The sample variance applies when data represents a sample taken
    from the relevant population. If it represents the entire population, you
    should use pvariance instead.
    """
    n, ss = _SS(data, m)
    if n < 2:
        raise StatsError('sample variance or standard deviation'
        ' requires at least two data points')
    return ss/(n-1)


def _SS(data, m):
    """SS = sum of square deviations.
    Helper function for calculating variance directly.
    """
    if m is None:
        # Two pass algorithm.
        data = as_sequence(data)
        m = mean(data)
    return _generalised_sum(data, lambda x: (x-m)**2)


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




if __name__ == '__main__':
    import doctest
    doctest.testmod()

