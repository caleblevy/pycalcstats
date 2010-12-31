#! /usr/bin/env python3

from distutils.core import setup

# Futz with the path so we can import metadata.
import sys
sys.path.insert(0, './src')
from stats import __version__, __author__, __author_email__

setup(
    name = "stats",
    package_dir={'': 'src'},
    py_modules = ['stats',],
    version = __version__,
    author = __author__,
    author_email = __author_email__,
    url = 'http://code.google.com/p/pycalcstats/',
    keywords = ["statistics", "mathematics", "calculator"],
    description = "Calculator-style statistical functions",
    long_description = """\
Statistical functions
---------------------

stats is a pure-Python module providing basic statistics functions similar
to those found on scientific calculators. It has over 40 statistics functions,
including:

Univariate statistics:
  * arithmetic, harmonic, geometric and quadratic means
  * median, mode, midrange, trimean
  * mean of angular quantities
  * running and weighted averages
  * quartiles, hinges and quantiles
  * variance and standard deviation (sample and population)
  * average deviation and median average deviation (MAD)
  * skew and kurtosis
  * standard error of the mean

Multivariate statistics:
  * Pearson's correlation coefficient
  * Q-correlation coefficient
  * covariance (sample and population)
  * linear regression
  * sums Sxx, Syy and Sxy

and others.


Requires Python 3.1 or better.
""",
    license = 'MIT',  # apologies for the American spelling
    classifiers = [
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.1",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        ],
    )

