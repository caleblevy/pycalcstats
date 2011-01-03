==============================
stats -- calculator statistics
==============================

Introduction
------------

stats is a pure-Python package providing statistics functions similar to
those found on scientific calculators. It has over 40 statistics functions,
including:

Basic calculator statistics:
  * arithmetic mean
  * variance (population and sample)
  * standard deviation (population and sample)

Univariate statistics:
  * harmonic, geometric and quadratic means
  * mode
  * mean of angular quantities
  * average deviation and median average deviation (MAD)
  * skewness and kurtosis
  * standard error of the mean

Order statistics:
  * median
  * quartiles, hinges and quantiles
  * range and midrange
  * interquartile range, midhinge and trimean
  * support for R-style quantile alternative calculation methods
  * Mathematica-style parameterized quantile calculation methods

Multivariate statistics:
  * Pearson's correlation coefficient
  * Q-correlation coefficient
  * covariance (sample and population)
  * linear regression
  * sums Sxx, Syy and Sxy

Coroutine versions of selected functions:
  * sum and mean
  * running and weighted averages
  * variance and standard deviation

among others.


Requires Python 3.1 or better.


Project home page
-----------------

http://code.google.com/p/pycalcstats/


Installation
------------

stats requires Python 3.1 or better. To install from source:

    1.  Download the stats .tar.gz file. (If you are reading this, you
        have probably already done that.)
    2.  Unpack the tarball:

        $ tar xf stats-XXX.tar.gz  # change XXX to the appropriate version
        $ cd stats-XXX/

    3. Run the installer:

        $ python3 setup.py install

The last step (running the installer) will need appropriate permissions to
succeed. You may need to run the installer as the root or Administrator user.


Usage
-----

An example of the basic calculator functionality:

    >>> import stats
    >>> stats.mean([1, 2, 3, 4, 5])
    3.0

A slightly more advanced example:

    >>> data = [1, 2, 3, 4, 5]
    >>> import stats
    >>> import stats.univar
    >>> s = stats.stdev(data)
    >>> stats.univar.stderrmean(s, len(data))
    1.234567


Licence
-------

stats is licenced under the MIT Licence. See the LICENCE.txt file
and the header of stats.__init__.py.


Self-test
---------

You can run the module's doctests by importing and executing the package
from the commandline:

    $ python3 -m stats

If all the doctests pass, no output will be printed. To get verbose output,
run with the -v switch:

    $ python3 -m stats -v


Known Issues
------------

See the CHANGES.txt file for a partial list of known issues and fixes. The
bug tracker is at http://code.google.com/p/pycalcstats/issues/list

