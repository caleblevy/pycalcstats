# ==============================================
# Private module -- do not import this directly.
# ==============================================


# This module contains private functions for calculating fractiles.
# Do NOT use anything in this module directly. Everything here is subject
# to change WITHOUT NOTICE.



# === Quartiles ===


class _Quartiles:
    """Private namespace for quartile calculation methods.

    Langford (2006) describes 15 methods for calculating quartiles, although
    some are mathematically equivalent to others:
        http://www.amstat.org/publications/jse/v14n3/langford.html

    We currently support the five methods described by Mathword and Dr Math:
        http://mathworld.wolfram.com/Quartile.html
        http://mathforum.org/library/drmath/view/60969.html
    plus Langford's Method #4 (CDF method).

    """
    def __new__(cls):
        raise RuntimeError('abstract namespace class')

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
        M = round((n+1)/2, EVEN)
        L = round((n+1)/4, UP)
        U = n+1-L
        assert U == round(3*(n+1)/4, DOWN)
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
        return (interpolate(data, L-1), interpolate(data, M-1),
        interpolate(data, U-1))

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
        return (interpolate(data, L-1), interpolate(data, M-1),
        interpolate(data, U-1))

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
        # the quartiles function in stats.py.

    # Lowercase aliases for the numeric method selectors for quartiles:
    QUARTILE_ALIASES = {
        'inclusive': 1,
        'tukey': 1,
        'hinges': 1,
        'exclusive': 2,
        'm&m': 2,
        'ti-85': 2,
        'm&s': 3,
        'minitab': 4,
        'f&p': 5,
        'excel': 5,
        'langford': 6,
        'cdf': 6,
        }


# === Quantiles (fractiles) ===


def cdf(data, p):
    """Langford's Method #4 for calculating general quantiles using the
    cumulative distribution function (CDF); this is also R's method 2 and
    SAS' method 5.
    """
    # Calculations are based on 1-based indexing.
    n = len(data)
    x = n*p
    i, f = int(x), x%1
    # Since x is non-negative x, i = floor(x). We actually want ceil(x)
    # instead, so we should add 1; however we would later subtract 1 to
    # account for 0-based indexing, so we do nothing.
    if f == 0:
        return (data[i] + data[i+1])/2
    else:
        # We're off by one here.
        return data[i-1]


def placeholder(data, p):
    pass

r1 = r3 = r4 = r5 = r6 = r7 = r8 = r9 = placeholder


# Numeric method selectors for quartiles. Numbers 1-9 MUST match the R
# calculation methods with the same number.
QUANTILE_MAP = {
    1: r1,
    2: cdf,
    3: r3,
    4: r4,
    5: r5,
    6: r6,
    7: r7,
    8: r8,
    9: r9,
    }
    # Note: if you add any additional methods to this, you must also update
    # the docstring for the quantiles function in stats.py.


# Lowercase aliases for the numeric method selectors for quantiles:
QUANTILE_ALIASES = {
    'sas-1': 4,
    'sas-2': 3,
    'sas-3': 1,
    'sas-4': 6,
    'sas-5': 2,
    'excel': 7,
    'cdf': 2,
    }


# === Helper functions ===

# Rounding modes:
UP = 0
DOWN = 1
EVEN = 2

def round(x, rounding_mode):
    """Round non-negative x, with ties rounding according to rounding_mode."""
    assert rounding_mode in (UP, DOWN, EVEN)
    assert x >= 0.0
    n, f = int(x), x%1
    if rounding_mode == UP:
        if f >= 0.5:
            return n+1
        else:
            return n
    elif rounding_mode == DOWN:
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


def interpolate(data, x):
    i, f = int(x), x%1
    if f:
        a, b = data[i], data[i+1]
        return a + f*(b-a)
    else:
        return data[i]

