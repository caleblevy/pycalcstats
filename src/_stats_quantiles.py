# ==============================================
# Private module -- do not import this directly.
# ==============================================



# Quantiles (fractiles), quartiles and related.
# ---------------------------------------------


def inclusive(data):
    """Return sample quartiles using Tukey's method."""
    # Returns the median, and the median of the two halves, including the
    # median in each half.
    n = len(data)
    rem = n%4
    a, m, b = n//4, n//2, (3*n)//4
    if rem == 0:
        q1 = (a-1, a)
        q2 = (m-1, m)
        q3 = (b-1, b)
    elif rem == 1:
        q1 = a
        q2 = m
        q3 = b
    elif rem == 2:
        q1 = a
        q2 = (m-1, m)
        q3 = b
    else:  # rem == 3
        q1 = (a-1, a)
        q2 = m
        q3 = (b-1, b)
    return (getitem(data, q1), getitem(data, q2), getitem(data, q3))


def exclusive(data):
    """Return sample quartiles using Moore and McCabe's method."""
    # Returns the median, and the median of the two halves, excluding the
    # median from each half. Also used by TI-85 calculators and newer models.
    n = len(data)
    rem = n%4
    a, m, b = n//4, n//2, (3*n)//4
    if rem == 0:
        q1 = (a-1, a)
        q2 = (m-1, m)
        q3 = (b-1, b)
    elif rem == 1:
        q1 = (a-1, a)
        q2 = m
        q3 = (b, b+1)  # b-1, b ?
    elif rem == 2:
        q1 = a
        q2 = (m-1, m)
        q3 = b
    else:  # rem == 3
        q1 = a
        q2 = m
        q3 = b
    return (getitem(data, q1), getitem(data, q2), getitem(data, q3))


def ms_quartile(data):
    """Return sample quartiles using Mendenhall and Sincich's method."""
    n = len(data)
    mid = n//2
    # Subtract 1 to adjust for zero-based indexing.
    lower = _round((n+1)/4, 'up') - 1
    upper = _round(3*(n+1)/4, 'down') - 1
    return (getitem(data, lower), getitem(data, mid), getitem(data, upper))


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



def getitem(data, idx):
    """Helper function for quantiles. Return the value of data at idx.

    If idx is an integer, returns data[idx].
    If idx is a two-tuple (a, b), returns the mean of data[a] and data[b].
    If idx is a float, returns the linear interpolation between data[n] and
    data[n+1], where n = int(idx).
    """
    if isinstance(idx, int):
        return data[idx]
    elif isinstance(idx, tuple):
        a, b = idx
        return (data[a] + data[b])/2
    elif isinstance(idx, float):
        n = int(idx)
        f = idx % 1
        x, y = data[n:n+1]
        return x + f*(y-x)



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

