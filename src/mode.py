
import operator


class ModeError(Exception):
    def __init__(self, *args):
        super().__init__(self, *args)
        


def modes(data, max_count=1):
    counts = collections.Counter(data)
    for d in data:
        counts[d] = counts.get(d, 0) + 1
    




def cmodes(data, size=5, count=1):
    """cmodes(data [, size [, count]] -> tuple of modes

    Return the estimated mode or modes for continuous data.

    Arguments:

        data    iterable of numeric data
        size    optional window size for estimating the mode (default=5)
        count   optional maximum number of modes to return (default=1)

    With continuous data, almost all data points are expected to be unique and
    so counting frequencies is unlikely to find the population mode. This
    estimates the mode (or modes) by averaging ``size`` data points at a time.

        .. Caution:: ``size`` must be at least 3 and should be as small as
           your data can bear. ... 

    If the number of modes actually found is larger than ``count``, raises
    ModeError
    """
    # Described as "estimating the rate of an inhomogeneous Poisson process
    # by Jth waiting time" by Press et al, "Numerical Recipes in Pascal",
    # Cambridge University Press, 1989, p.508.
    if not isinstance(size, int):
        raise TypeError(
            'expected size to be an int but got %s' % type(size).__name__
            )
    if not isinstance(count, int):
        raise TypeError(
            'expected count to be an int but got %s' % type(size).__name__
            )
    data = sorted(data)
    if size < 3:
        raise ValueError('window size must be no less than 3')
    elif size > len(data):
        raise ValueError(
            'window size must be no greater than the number of data points'
            )
    if count <= 0:
        raise ValueError('count must be at least 1')
    elif count > len(data):
        raise ValueError('count must be no more than the number of data points')
    # Create a probability table.
    table = []
    Y = size/len(data)
    for i in range(len(data)-size):
        xi = data[i]
        xij = data[i+size]
        x = (xi+xij)/2  # Average of the points within window size.
        p = Y/(xij-xi)  # Probability of seeing that (average) x.
        table.append((p,q))
    table.sort(key=operator.itemgetter(1), reverse=True)
    if count == 1:


