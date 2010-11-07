# ==============================================
# Private module -- do not import this directly.
# ==============================================



# Quantiles (fractiles), quartiles and related.
# ---------------------------------------------


# These functions that follow all assume that the data is a sorted
# list of length at least 3. No error checking is performed.


def inclusive(data):
    """Return sample quartiles using Tukey's method."""
    # Returns the median, and the median of the two halves, including the
    # median in each half.
    n = len(data)
    rem = n%4
    a, m, b = n//4, n//2, (3*n)//4
    if rem == 0:
        q1 = (data[a-1] + data[a])/2
        q2 = (data[m-1] + data[m])/2
        q3 = (data[b-1] + data[b])/2
    elif rem == 1:
        q1 = data[a]
        q2 = data[m]
        q3 = data[b]
    elif rem == 2:
        q1 = data[a]
        q2 = (data[m-1] + data[m])/2
        q3 = data[b]
    else:  # rem == 3
        q1 = (data[a-1] + data[a])/2
        q2 = data[m]
        q3 = (data[b-1] + data[b])/2
    return (q1, q2, q3)


def exclusive(data):
    """Return sample quartiles using Moore and McCabe's method."""
    # Returns the median, and the median of the two halves, excluding the
    # median from each half. Also used by TI-85 calculators and newer models.
    n = len(data)
    rem = n%4
    a, m, b = n//4, n//2, (3*n)//4
    if rem == 0:
        q1 = (data[a-1] + data[a])/2
        q2 = (data[m-1] + data[m])/2
        q3 = (data[b-1] + data[b])/2
    elif rem == 1:
        q1 = (data[a-1] + data[a])/2
        q2 = data[m]
        q3 = (data[b] + data[b+1])/2
    elif rem == 2:
        q1 = data[a]
        q2 = (data[m-1] + data[m])/2
        q3 = data[b]
    else:  # rem == 3
        q1 = data[a]
        q2 = data[m]
        q3 = data[b]
    return (q1, q2, q3)


def ms_quartile(data):
    """Return sample quartiles using Mendenhall and Sincich's method."""
    n = len(data)
    midpoint = n//2
    lower = _round((n+1)/4, 'up')
    upper = _round(3*(n+1)/4, 'down')
    # Subtract 1 to adjust for zero-based indexing.
    return (data[lower-1], data[midpoint], data[upper-1])


def minitab(data):
    """Return sample quartiles using the method used by Minitab."""
    raise NotImplementedError


def excel(data):
    """Return sample quartiles using the method used by Excel."""
    # Method recommended by Freund and Perles.
    raise NotImplementedError




# Note: if you modify this, you must also update the docstring for
# the quartiles function in stats.py.
MAP = {
    0: inclusive,   't': inclusive,
    1: exclusive,   'm&m': exclusive,
                    'ti-85': exclusive,
    2: ms_quartile, 'm&s': ms_quartile,
    3: minitab,     'mt': minitab,
                    'minitab': minitab,
    4: excel,       'fp':excel,
                    'excel': excel,
    }




def _round(x, direction):
    """Round non-negative x with direction (up, down) on ties."""
    direction = direction.lower()
    assert direction in ('up', 'down')
    assert x >= 0.0
    if direction == 'up':
        if x%1 >= 0.5:
            return int(x) + 1
        else:
            return int(x)
    else:
        if x%1 > 0.5:
            return int(x) + 1
        else:
            return int(x)

