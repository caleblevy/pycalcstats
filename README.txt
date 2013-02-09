==============================
stats -- calculator statistics
==============================

Introduction
------------

stats is a pure-Python package providing simple statistics functions 
similar to those found on scientific calculators, including:

Basic calculator statistics:
  * arithmetic mean
  * variance (population and sample)
  * standard deviation (population and sample)


Requires Python 3.3 or better.


Project home page
-----------------

http://code.google.com/p/pycalcstats/


Usage
-----

An example of the basic calculator functionality:

    >>> import stats
    >>> stats.mean([1, 2, 3, 4, 5])
    3.0


Licence
-------

stats is licenced under the MIT Licence. See the LICENCE.txt file
and the header of stats.py.


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

