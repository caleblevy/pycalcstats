#! /usr/bin/env python3

from distutils.core import setup

setup(  name="stats.py",
	version="0.1.1a",
    	description="Elementary statistics module for Python",
	author="Steven D'Aprano",
	author_email="steve+python@pearwood.info",
	url="http://pypi.python.org/pypi/stats/0.1a",
	package_dir={'': 'src'},
        packages=['stats'],
        long_description = """\
Statistical functions
---------------------

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

Requires Python 2.5, 2.6, 2.7 or 3.1.
""",
        license = 'MIT',  # apologies for the American spelling
        classifiers = [
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.5",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.1",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        ],
    )
