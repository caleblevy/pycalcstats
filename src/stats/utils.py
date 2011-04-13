#!/usr/bin/env python3

##  Copyright (c) 2011 Steven D'Aprano.
##  See the file __init__.py for the licence terms for this software.

"""
General utilities used by the stats package.
"""


__all__ = ['add_partial', 'coroutine','minmax']


import collections
import functools
import itertools
import math


# === Exceptions ===

class StatsError(ValueError):
    pass


# === Helper functions ===

def sorted_data(func):
    """Decorator to sort data passed to stats functions."""
    @functools.wraps(func)
    def inner(data, *args, **kwargs):
        data = sorted(data)
        return func(data, *args, **kwargs)
    return inner


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


def as_sequence(iterable):
    """Helper function to convert iterable arguments into sequences."""
    if isinstance(iterable, (list, tuple)): return iterable
    else: return list(iterable)


def coroutine(func):
    """Decorator to prime coroutines when they are initialised."""
    @functools.wraps(func)
    def started(*args, **kwargs):
        cr = func(*args,**kwargs)
        cr.send(None)
        return cr
    return started


# Modified from http://code.activestate.com/recipes/393090/
# Thanks to Raymond Hettinger.
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


def _generalised_sum(data, func):
    """_generalised_sum(data, func) -> len(data), sum(func(items of data))

    Return a two-tuple of the length of data and the sum of func() of the
    items of data. If func is None, use just the sum of items of data.
    """
    # Try fast path.
    try:
        count = len(data)
    except TypeError:
        # Slow path for iterables without len.
        # We want to support BIG data streams, so avoid converting to a
        # list. Since we need both a count and a sum, we iterate over the
        # items and emulate math.fsum ourselves.
        ap = add_partial
        partials = []
        count = 0
        if func is None:
            # Note: we could check for func is None inside the loop. That
            # is much slower. We could also say func = lambda x: x, which
            # isn't as bad but still somewhat expensive.
            for count, x in enumerate(data, 1):
                ap(x, partials)
        else:
            for count, x in enumerate(data, 1):
                ap(func(x), partials)
        total = math.fsum(partials)
    else: # Fast path continues.
        if func is None:
            # See comment above.
            total = math.fsum(data)
        else:
            total = math.fsum(func(x) for x in data)
    return count, total
    # FIXME this may not be accurate enough for 2nd moments (x-m)**2
    # A more accurate algorithm may be the compensated version:
    #   sum2 = sum(x-m)**2) as above
    #   sumc = sum(x-m)  # Should be zero, but may not be.
    #   total = sum2 - sumc**2/n


def _sum_sq_deviations(data, m):
    """Returns the sum of square deviations (SS).
    Helper function for calculating variance.
    """
    if m is None:
        # Two pass algorithm.
        data = as_sequence(data)
        n, total = _generalised_sum(data, None)
        if n == 0:
            return (0, total)
        m = total/n
    return _generalised_sum(data, lambda x: (x-m)**2)


def _sum_prod_deviations(xydata, mx, my):
    """Returns the sum of the product of deviations (SP).
    Helper function for calculating covariance.
    """
    if mx is None:
        # Two pass algorithm.
        xydata = as_sequence(xydata)
        nx, sumx = _generalised_sum((t[0] for t in xydata), None)
        if nx == 0:
            raise StatsError('no data items')
        mx = sumx/nx
    if my is None:
        # Two pass algorithm.
        xydata = as_sequence(xydata)
        ny, sumy = _generalised_sum((t[1] for t in xydata), None)
        if ny == 0:
            raise StatsError('no data items')
        my = sumy/ny
    return _generalised_sum(xydata, lambda t: (t[0]-mx)*(t[1]-my))


def _validate_int(n):
    # This will raise TypeError, OverflowError (for infinities) or
    # ValueError (for NANs or non-integer numbers).
    if n != int(n):
        raise ValueError('requires integer value')


# Rounding modes.
_UP, _DOWN, _EVEN = 0, 1, 2

def _round(x, rounding_mode):
    """Round non-negative x, with ties rounding according to rounding_mode."""
    assert rounding_mode in (_UP, _DOWN, _EVEN)
    assert x >= 0.0
    n, f = int(x), x%1
    if rounding_mode == _UP:
        if f >= 0.5:
            return n+1
        else:
            return n
    elif rounding_mode == _DOWN:
        if f > 0.5:
            return n+1
        else:
            return n
    else:
        # Banker's rounding to EVEN.
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


def _interpolate(data, x):
    i, f = math.floor(x), x%1
    if f:
        a, b = data[i], data[i+1]
        return a + f*(b-a)
    else:
        return data[i]


# === Generic utilities ===

from stats import minmax
