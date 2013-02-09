#!/usr/bin/env python3

##  Module stats.py
##
##  Copyright (c) 2013 Steven D'Aprano.
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


"""Simple statistics for Python 3.

The ``stats`` module provides nine statistics functions:

    Function        Description
    ==============  =============================================
    mean            Arithmetic mean (average) of data.
    minmax          Minimum and maximum of the arguments.
    product         Product of data.
    pstdev          Population standard deviation of data.
    pvariance       Population variance of data.
    running_product Running product coroutine.
    running_sum     High-precision running sum coroutine.
    stdev           Sample standard deviation of data.
    sum             High-precision sum of data.
    variance        Sample variance of data (bias-corrected).

The module also includes two public utility functions plus an exception
class used for some statistical errors:

    Name            Description
    ==============  =============================================
    add_partial     Utility for performing high-precision sums.
    coroutine       Utility for initialising coroutines.
    StatsError      Subclass of ValueError.


Examples
--------

>>> import stats
>>> stats.mean([-1.0, 2.5, 3.25, 5.75])
2.625
>>> stats.stdev([2.5, 3.25, 5.5, 11.25, 11.75])  #doctest: +ELLIPSIS
4.38961843444...
>>> stats.minmax(iter([19, 23, 15, 42, 31]))
(15, 42)


"""


# === Module metadata ===

__version__ = "0.2.0a"
__date__ = "2013-??????"  # FIXME 
__author__ = "Steven D'Aprano"
__author_email__ = "steve+python@pearwood.info"


__all__ = [ 'StatsError', 'add_partial',
            'coroutine', 'mean', 'minmax', 'product',
            'pstdev', 'pvariance', 'running_sum', 'stdev',
            'sum', 'variance',
          ]


# === Imports ===

import collections
import functools
import itertools
import math
import operator

from builtins import sum as _sum



# === Exceptions ===

class StatsError(ValueError):
    pass


# === Utility functions ===

# Modified from http://code.activestate.com/recipes/393090/
# Thanks to Raymond Hettinger.
def add_partial(x, partials):
    """Helper function for full-precision summation of binary floats.

    Adds finite (not NAN or INF) x in place to the list partials.

    Example usage:

    >>> partials = []
    >>> add_partial(1e100, partials)
    >>> add_partial(1e-100, partials)
    >>> add_partial(-1e100, partials)
    >>> partials
    [1e-100, 0.0]

    Initialise partials to be a list containing at most one finite float
    (i.e. no INFs or NANs). Then for each float you wish to add, call
    ``add_partial(x, partials)``.

    When you are done, call sum(partials) to round the summation to the
    precision supported by float.

    If you initialise partials with more than one value, or with special
    values (NANs or INFs), results are undefined.
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






