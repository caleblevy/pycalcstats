from distutils.core import setup

from stats import __version__, __author__, __author_email__

setup(
    name = "stats",
    py_modules=["stats"],
    version = __version__,
    author = __author__,
    author_email = __author_email__,
    url = 'http://pypi.python.org/pypi/stats',
    keywords = ["statistics", "mathematics", "calculator"],
    description = "Calculator-style statistical functions",
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

Requires Python 3.1.
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

