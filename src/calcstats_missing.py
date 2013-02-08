##  Copyright (c) 2011 Steven D'Aprano.

"""Calculator statistics functions with MISSING values.

"""

import collections
import functools
import itertools
import math
import numbers
import operator

from builtins import sum as _builtin_sum


# Package metadata.
__version__ = "0.2.0a"
__date__ = "2011-08-19"
__author__ = "Steven D'Aprano"
__author_email__ = "steve+python@pearwood.info"

__all__ = [ 'add_partial', 'count', 'mean', 'minmax', 'MISSING', 'product',
            'pstdev', 'pvariance', 'StatsError', 'stdev', 'sum', 'variance',
          ]


# === Exceptions ===

class StatsError(ValueError):
    pass


class _EmptyIterable(ValueError):
    pass


# === Flags ===

# Two flags are currently defined. There may be more later.

SKIP_MISSING = 1  # Ignore missing values in data.
PREFER_NANS = 2  # Return NANs instead of raising an exception when appropriate.

DEFAULT_FLAGS = SKIP_MISSING


# === Support for missing data ===

class MissingType:
    """Singleton missing value MISSING.

    Simple arithmetic operations on this will return itself:

    >>> MISSING + 42
    MISSING

    """
    __slots__ = []
    _inst = None

    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
        return cls._inst

    def __str__(self):
        return "MISSING"
    __repr__ = __str__

    def _assimilate(self, *args):
        # We are the Borg, prepare to be assimilated.
        return self

    __add__ = __sub__ = __mul__ = __truediv__ = __floordiv__ \
      = __radd__ = __rsub__ = __rmul__ = __rtruediv__ = __rfloordiv__ \
      = __divmod__ = __mod__ = __pow__ = __rdivmod__ = __rmod__ = __rpow__ \
      = __and__ = __or__ = __xor__ = __rand__ = __ror__ = __rxor__ \
      = __lshift__ = __rshift__ = __rlshift__ = __rrshift__ \
      = __abs__ = __ceil__ = __floor__ = __invert__ = __neg__ = __pos__ \
      = __round__ = __trunc__ = _assimilate

    def __bool__(self): return False

    # Comparisons. Equal to itself, less than everything else.
    def __lt__(self, other): return False if other is self else True
    def __le__(self, other): return True
    def __gt__(self, other): return False
    def __ge__(self, other): return True if other is self else False

MISSING = MissingType()
del MissingType

check_missing = False  # By default, we don't ignore missing values.

not_missing = functools.partial(operator.is_not, MISSING)


# === Utilities ===

try:
    from math import isfinite as _isfinite
    def isspecial(x):
        return x is MISSING or not _isfinite(x)
except ImportError:
    from math import isnan as _isnan, isinf as _isinf
    def isspecial(x):
        return x is MISSING or _isnan(x) or _isinf(x)

isspecial.__doc__ = """\
isspecial(x) -> True|False

Return True if x is MISSING, infinite or a NAN, otherwise return False.
"""


def _isiterable(obj):
    """Return True if obj is an iterable, otherwise False.

    >> _isiterable([42, 23])
    True
    >> _isiterable(42)
    False

    """
    try:
        iter(obj)
    except TypeError:
        return False
    else:
        return True


def _peek(iterable):
    """Peek ahead of an iterable.

    If iterable is a non-empty sequence, returns a two tuple:
        (first item, sequence)

    >>> _peek([1, 2, 3])
    (1, [1, 2, 3])

    If iterable is a non-empty iterator, returns a two tuple:
        (first item, iterator including first item)

    >>> a, b = _peek([1, 2, 3])
    >>> a == 1 and list(b) == [1, 2, 3]
    True

    Note that the returned iterator is not necessarily the same iterator
    object as that being peeked at.

    If iterable is empty, raises _EmptyIterable.
    """
    if isinstance(iterable, collections.Sequence):
        if iterable:
            return (iterable[0], iterable)
    else:
        try:
            first = next(iterable)
        except StopIteration:
            pass
        else:
            it = itertools.chain([first], iterable)
            return (first, it)
    raise _EmptyIterable


def _adder(x, partials):
    """Add numeric x or MISSING to partials.

    x:          a number, infinity, NAN or MISSING.
    partials:   a list containing either a single special value or zero or
                more partial sums.

    partials is modified in place.

    partials should be initialised to an empty list or a list of one finite
    number (not a NAN or INF). If partials is not initialised correctly,
    behaviour is undefined.

    """
    if isspecial(x):
        _add_special(x, partials)
    else:
        try:
            _add_partial(x, partials)
        except TypeError:
            if not isinstance(x, numbers.Real):
                raise
            # Otherwise probably trying to e.g. add Decimal to float.
            # Downgrade everything to float and try again.
            partials[:] = map(float, partials)
            x = float(x)
            _add_partial(x, partials)


def _add_partial(x, partials):
    """Helper function for full-precision summation.

    Adds finite (not NAN or INF) x in place to the list partials.

    Example usage:

    >>> partials = []
    >>> _add_partial(1e100, partials)
    >>> _add_partial(1e-100, partials)
    >>> _add_partial(-1e100, partials)
    >>> partials
    [1e-100, 0.0]

    Initialise partials to be a list containing at most one finite number
    (i.e. no INFs or NANs). Then for each number you wish to add, call
    ``add_partial(x, partials)``.

    When you are done, call built-in sum(partials) to round the summation
    to the precision supported by float.

    Also works with other numeric values such as ints, Fractions, etc.

    If you initialise partials with more than one value, or with special
    values (NANs or INFs), results are undefined.
    """
    # Modified from Raymond Hettinger's recipe
    # http://code.activestate.com/recipes/393090/
    assert not isspecial(x), 'non-finite valued argument'
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
    # Rounded x+y stored in hi with the round-off stored in lo.  Together
    # hi+lo are exactly equal to x+y.  The inner loop applies hi/lo summation
    # to each partial so that the list of partial sums remains exact.
    # Depends on IEEE-754 arithmetic guarantees.  See proof of correctness at:
    # www-2.cs.cmu.edu/afs/cs/project/quake/public/papers/robust-arithmetic.ps


def _add_special(x, partials):
    """Helper function for summation of special values.

    Adds non-finite (NAN, INF or MISSING) x in place to the list partials.

    MISSING + anything = MISSING
    NAN + anything except MISSING = NAN
    INF + -INF = NAN
    INF + anything else = INF
    -INF + anything else = -INF

    Addition of Decimal NANs and INFs depend on the context, and may raise
    an exception.
    """
    assert isspecial(x), 'non-special valued argument'
    if x is MISSING:
        partials[:] = [x]
    elif len(partials) != 1:
        partials[:] = [x]
    else:
        y = partials[0]
        if y is MISSING:
            return
        try:
            z = y + x  # Note that x should be the right hand operand. This
            # usually won't make a difference, but may for user-defined types.
        except TypeError:
            # Probably trying to add Decimal to float, or similar.
            # Downgrade everything to float.
            z = float(y) + float(x)
        # Other exceptions (e.g. decimal.InvalidOperation or similar) should
        # be allowed to go through.
    assert len(partials) == 1


def _count_row(row, counts, skip_missing):
    delta = len(row) - len(counts)
    if delta > 0:
        counts.extend([0]*delta)
    if skip_missing:
        for i,x in enumerate(row):
            if x is not MISSING:
                counts[i] += 1
    else:
        for i in range(len(row)):
            counts[i] += 1
    return counts


def _add_row(row, totals, skip_missing):
    delta = len(row) - len(totals)
    if delta > 0:
        totals.extend([[] for _ in range(delta)])
    if skip_missing:
        for i,x in enumerate(row):
            if x is not MISSING:
                _adder(x, totals[i])
    else:
        for i,x in enumerate(row):
            _adder(x, totals[i])


def count(data, *, flags=None):
    """Count values in data.

    Keyword-only argument flags specifies behaviour of the count function. If
    the SKIP_MISSING bit of flags is set, missing values are ignored. Missing
    values must be the global MISSING. Otherwise, MISSING is counted as a
    regular value. If flags is None or not given, it defaults to the global
    variable DEFAULT_FLAGS.

    If the values in data are scalars, count returns a single value:

    >>> data = [1, 2, 3, MISSING, 5]
    >>> count(data, flags=SKIP_MISSING)
    4
    >>> count(data, flags=0)
    5

    If the values are sequences, they are treated as rows and the count of
    each column is returned:

    >>> data = [[1, 2, 3, 4, 5],
    ...         [1, 1, 1      ],
    ...         [2, 4, 6, 8   ]]
    >>> count(data)
    [3, 3, 3, 2, 1]

    """
    if flags is None:
        flags = DEFAULT_FLAGS
    try:
        first, data = _peek(data)
    except _EmptyIterable:
        return 0
    if isinstance(first, collections.Sequence):
        count = [0]*len(first)
        for row in data:
            count = _count_row(row, count, flags&SKIP_MISSING)
    else:
        if flags&SKIP_MISSING:
            try:
                count = len(data) - data.count(MISSING)
            except (TypeError, AttributeError):
                count = _builtin_sum(1 for x in data if x is not MISSING)
        else:
            try:
                count = len(data)
            except TypeError:
                count = _builtin_sum(1 for _ in data)
    return count


def _len_sum(data, skip_missing):
    """Return the high-precision sum of iterable of numbers.

    Supports int, fraction, float, Decimal, and mixes of same. Tries to
    keep consistent types (e.g. sum of fractions is a fraction) but will
    downgrade to float if necessary. Sums of fractions and ints are exact,
    while Decimal and float include rounding error.

    If skip_missing is true, values which are the global MISSING are ignored.
    Otherwise, if iterable includes a MISSING value, the sum itself is also
    MISSING.

    Other special values (float and Decimal NANs and INFs) are treated
    appropriately.
    """
    try:
        first, data = _peek(data)
    except EmptyIterable:
        return (0, 0)
    if isinstance(first, collections.Sequence):
        total = [[] for _ in range(len(first))]
        count = [0]*len(first)
        for row in data:
            count = _count_row(row, count, skip_missing)
            _add_row(row, total, skip_missing)
        total = [t[0] if len(t)==1 else _builtin_sum(t) for t in total]
    else:
        total = []
        count = 0
        if skip_missing:
            for x in data:
                if x is not MISSING:
                    count += 1
                    _adder(x, total)
        else:
            for x in data:
                count += 1
                _adder(x, total)
        total = total[0] if len(total)==1 else _builtin_sum(total)
    return (count, total)


def sum(data, start=0, *, flags=None):
    """Sum the values in data, adding start.

    Keyword-only argument flags specifies behaviour of the sum function. If
    flags is None or not given, it defaults to the global variable
    DEFAULT_FLAGS.

    If the values in data are scalars, ``sum`` returns a single value:

    >>> data = [1, 2, 3, MISSING, 5]
    >>> sum(data, flags=SKIP_MISSING)
    11
    >>> sum(data, 1000, flags=SKIP_MISSING)
    1011

    Unless skipped, the sum of data containing MISSING is itself MISSING:

    >>> count(data, flags=0)
    MISSING

    If the values are sequences, they are treated as rows and the count of
    each column is returned:

    >>> data = [[1, 2, 3, 4, 5],
    ...         [1, 1, 1      ],
    ...         [1, 2, 4, 6   ]]
    >>> sum(data)
    [3, 5, 8, 10, 5]

    Note that blank columns in a row only the specific MISSING if the rows differ

    """
    if flags is None:
        flags = DEFAULT_FLAGS
    _, total = _len_sum(data, flags&SKIP_MISSING)


def _count_sum(data, *, missing=None):
    """Return (count(data), sum(data)) in effectively a single pass."""
    if missing is None:
        missing = check_missing
    # Optimize lists and other sequences as a fast and common case.
    if isinstance(data, collections.Sequence):
        if data and isinstance(data[0], float):
            # If the first item is a float, we assume the rest are also
            # floats or compatible with floats, and use a fast, high-accuracy
            # sum function.
            adder = math.fsum
        else:
            # Fall back on a slower sum function. This should (more or less)
            # emulate math.fsum, but not coerce arguments into floats unless
            # needed.
            adder = _sum
        n = len(data)
        if missing:
            num_missing = data.count(MISSING)
            n -= num_missing
            data = filter(not_missing, data)
        # Easiest way to check for bad data is to actually attempt the sum
        # and allow any exceptions to propogate.
        total = adder(data)
        if num_missing:
            # 
    else:
        # Handle arbitrary iterables.
        if missing:
            data = filter(not_missing, data)
        data = _countiter(data)
        assert data.count == 0
        total = _sum(data)
        n = data.count
    assert n >= 0
    return (n, total)


def _variance(data, mu, p, *, missing=None):
    """Return an estimate of variance relative to population mean mu using N-p degrees of freedom."""
    if missing is None:
        missing = check_missing
    if mu is None:
        # First pass over data to calculate the mean. If it's not a sequence,
        # make it so first.
        if missing:
            data = list(filter(not_missing, data))
            missing = False  # Won't need this later.
        elif not isinstance(data, list):
            data = list(data)
        mu = mean(data, missing=False)
    # Iterate over the data calculating the sum of squares of residuals.
    # Optimize the case where data looks like a list of floats.
    if (isinstance(data, collections.Sequence) and data and
        isinstance(data[0], float)):
        if missing:
            data = list(filter(not_missing, data))
        n = len(data)
        sum_squares = math.fsum((x-mu)**2 for x in data)
        sum_residues = math.fsum(x-mu for x in data)
        # sum_residues should be zero, if floats were infinitely precise. But
        # because they aren't, we use it to calculate the compensated sum of
        # squares which should be more accurate than sum_squares alone.
    else:
        ap = add_partial
        n = 0
        sum_squares = []
        sum_residues = []
        if missing:
            data = filter(not_missing, data)
        for x in data:
            n += 1
            residue = x-mu
            ap(residue**2, sum_squares)
            ap(residue, sum_residues)
        sum_squares = _builtin_sum(sum_squares)
        sum_residues = _builtin_sum(sum_residues)
    total = sum_squares - sum_residues**2/n
    assert n >= 0
    assert sum_squares >= 0
    if n <= p:
        raise StatsError(
            'at least %d items are required but only got %d' % (p+1, n))
    return sum_squares/(n-p)


# === Statistics functions ===

def count(data, *, missing=None):
    if missing is None:
        missing = check_missing
    # Optimize lists as a fast and common case.
    if isinstance(data, list):
        n = len(data)
        if missing:
            n -= data.count(MISSING)
    else:
        # Handle arbitrary iterables.
        if missing:
            data = filter(not_missing, data)
        n = sum(1 for x in data)
    assert n >= 0
    return n


def product(data, start=1, *, missing=None):
    if missing is None:
        missing = check_missing
    if missing:
        data = filter(not_missing, data)
    return functools.reduce(operator.mul, data, start)
    # Don't be tempted to do something clever with the sum of logarithms,
    # because it is far less accurate than a simple-minded multiplication.


def sum(data, start=0, *, missing=None):
    if missing is None:
        missing = check_missing
    # Optimize lists of floats as a fast and common case.
    if (isinstance(data, collections.Sequence) and data and
        isinstance(data[0], float)):
        # If the first item is a float, we assume the rest are also floats or
        # compatible with floats, and use a fast, high-accuracy sum function.
        adder = math.fsum
    else:
        # Fall back on a slower sum function. This should (more or less)
        # emulate math.fsum, but not coerce arguments into floats unless
        # absolutely necessary.
        adder = _sum
    if missing:
        data = filter(not_missing, data)
    return start + adder(data)


def mean(data, *, missing=None):
    n, total = _count_sum(data, missing=missing)
    if not n:
        raise StatsError('mean of empty sequence is not defined')
    return total/n


def pvariance(data, mu=None, *, missing=None):
    return _variance(data, mu, 0, missing=missing)


def pstdev(data, mu=None, *, missing=None):
    ss = pvariance(data, mu, missing=missing)
    return ss if ss is MISSING else math.sqrt(ss)


def variance(data, mu=None, *, missing=None):
    return _variance(data, mu, 1, missing=missing)


def stdev(data, mu=None, *, missing=None):
    ss = variance(data, mu, missing=missing)
    return ss if ss is MISSING else math.sqrt(ss)


def minmax(*values, **kw):
    """minmax(iterable [, key=func]) -> (minimum, maximum)
    minmax(a, b, c, ... [, key=func]) -> (minimum, maximum)

    With a single iterable argument, return a two-tuple of its smallest and
    largest items. With two or more arguments, return the smallest and
    largest arguments. ``minmax`` is similar to the built-ins ``min`` and
    ``max``, but can return the two items with a single pass over the data,
    allowing it to work with iterators.

    >>> minmax([3, 2, 1, 6, 5, 4])
    (1, 6)
    >>> minmax(4, 5, 6, 1, 2, 3)
    (1, 6)

    The optional keyword-only argument ``key`` specifies a key function:

    >>> minmax('aaa', 'bbbb', 'c', 'dd', key=len)
    ('c', 'bbbb')

    """
    if len(values) == 0:
        raise TypeError('minmax expected at least one argument, but got none')
    elif len(values) == 1:
        values = values[0]
    if list(kw.keys()) not in ([], ['key']):
        raise TypeError('minmax received an unexpected keyword argument')
    if isinstance(values, collections.Sequence):
        # For speed, fall back on built-in min and max functions when
        # data is a sequence and can be safely iterated over twice.
        minimum = min(values, **kw)
        maximum = max(values, **kw)
        # The number of comparisons is N-1 for both min() and max(), so the
        # total used here is 2N-2, but performed in fast C.
    else:
        # Iterator argument, so fall back on a (slow) pure-Python solution
        # that calculates the min and max lazily. Even if values is huge,
        # this should work.
        # Note that the number of comparisons is 3*ceil(N/2), which is
        # approximately 50% fewer than used by separate calls to min & max.
        key = kw.get('key')
        if key is not None:
            it = ((key(value), value) for value in values)
        else:
            it = ((value, value) for value in values)
        try:
            keyed_min, minimum = next(it)
            keyed_max, maximum = keyed_min, minimum
        except StopIteration:
            # Don't directly raise an exception inside the except block,
            # as that exposes the StopIteration to the caller. That's an
            # implementation detail that should not be exposed. See PEP 3134
            # http://www.python.org/dev/peps/pep-3134/
            # and specifically the open issue "Suppressing context".
            empty = True
        else:
            empty = False
        if empty:
            raise ValueError('minmax argument is empty')
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


