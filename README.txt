==============================
stats -- calculator statistics
==============================

Introduction
------------

stats is a pure-Python module providing basic statistics functions
similar to those found on scientific calculators. It currently includes:

Univariate statistics including:
* arithmetic, harmonic, geometric and quadratic means
* median, mode
* standard deviation and variance (sample and population)

Multivariate statistics including:
* Pearson's correlation coefficient
* covariance (sample and population)
* linear regression

and others.


Requires Python 3.1 or better. It may work with Python 3.0, but that is
unsupported and untested.


Project home page
-----------------

http://code.google.com/p/pycalcstats/


Installation
------------

stats requires Python 3.1. To install, do the usual:

    python3 setup.py install


Licence
-------

stats is licenced under the MIT Licence. See the LICENCE.txt file
and the header of stats.py.


Self-test
---------

You can run the module's doctests by executing the file from the commandline.
On most Linux or UNIX systems:

    $ chmod u+x stats.py  # this is only needed once
    $ ./stats.py

or:

    $ python3 stats.py

If all the doctests pass, no output will be printed. To get verbose output,
run with the -v switch:

    $ python3 stats.py -v


Known Issues
------------

See the CHANGES.txt file for a partial list of known issues and fixes. The
bug tracker is at http://code.google.com/p/pycalcstats/issues/list

