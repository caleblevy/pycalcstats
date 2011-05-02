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


def as_sequence(iterable):
    """Helper function to convert iterable arguments into sequences."""
    if isinstance(iterable, (list, tuple)): return iterable
    else: return list(iterable)


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



# === Generic utilities ===

from stats import minmax, add_partial
