#!/usr/bin/env python3

##  Copyright (c) 2011 Steven D'Aprano.
##  See the file __init__.py for the licence terms for this software.

"""
Run the stats package as if it were an executable module.

Usage:
    $ python3 -m stats [options]

Options:
    -h  --help      Print this help text.
    -V  --version   Print the version number.
    -v  --verbose   Run tests verbosely.
    -q  --quiet     Don't print anything on success.

With no options, perform a self-test of the stats package by running all
doctests in the package. By default, failed tests will be printed. If all
tests pass, a count of how many tests were performed is printed.

To print details of all tests regardless of whether they succeed or fail,
pass the verbose flag after the package name:

    $ python3 -m stats -v

To suppress output if all tests pass, pass the quiet flag:

    $ python3 -m stats -q

"""
import sys

def process_options():
    argv = sys.argv[1:]
    if '-h' in argv or '--help' in argv:
        print(__doc__)
        sys.exit(0)
    verbose = '-v' in argv or '--verbose' in argv
    quiet = '-q' in argv or '--quiet' in argv
    if verbose and quiet:
        print('cannot be both quiet and verbose', file=sys.stderr)
        sys.exit(1)
    if '-V' in argv or '--version' in argv:
        import stats
        print(stats.__version__)
        sys.exit(0)
    return verbose, quiet


def self_test(verbose, quiet):
    assert not (verbose and quiet)
    import doctest
    import stats, stats.co, stats.multivar, stats.order, \
           stats.univar, stats.utils
    modules = (stats, stats.co, stats.multivar, stats.order,
               stats.univar, stats.utils
               )
    failed = tried = 0
    for module in modules:
        a, b = doctest.testmod(module, verbose=verbose)
        failed += a
        tried += b
    if failed == 0 and not quiet:
        print("Successfully run %d doctests from %d files."
              % (tried, len(modules)))
    return failed


if __name__ == '__main__' and __package__ is not None:
    verbose, quiet = process_options()
    sys.exit(self_test(verbose, quiet))

