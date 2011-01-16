#!/usr/bin/env python3

##  Copyright (c) 2011 Steven D'Aprano.
##  See the file stats/__init__.py for the licence terms for this software.

"""
Test suite for the stats package.

"""

# Note: do not use self.fail... unit tests, as they are deprecated in
# Python 3.2. Although plural test cases such as self.testEquals and
# friends are not officially deprecated, they are discouraged.

import unittest

import collections


def approx_equal(x, y, tol=1e-12, rel=1e-7):
    if tol is rel is None:
        # Fall back on exact equality.
        return x == y
    tests = []
    if tol is not None: tests.append(tol)
    if rel is not None: tests.append(rel*abs(x))
    assert tests
    return abs(x - y) <= max(tests)


USE_DEFAULT = object()
class NumericTestCase(unittest.TestCase):
    tol = None
    rel = 1e-9
    def assertApproxEqual(
        self, actual, expected, tol=USE_DEFAULT, rel=USE_DEFAULT, msg=None
        ):
        # Note that unlike many other unittest assert* methods, this
        # is asymmetric -- the first argument is treated differently from
        # the second.
        if tol is USE_DEFAULT: tol = self.tol
        if rel is USE_DEFAULT: rel = self.rel
        if (isinstance(actual, collections.Sequence) and
        isinstance(expected, collections.Sequence)):
            result = self._check_approx_seq(actual, expected, tol, rel, msg)
        else:
            result = self._check_approx_num(actual, expected, tol, rel, msg)
        if result:
            raise result

    def _check_approx_seq(self, actual, expected, tol, rel, msg):
        if len(actual) != len(expected):
            standardMsg = (
                "actual and expected sequences differ in length; expected"
                " %d items but found %d." % (len(expected), len(actual)))
            msg = self._formatMessage(msg, standardMsg)
            return self.failureException(msg)  # Don't raise.
        for i, (a,e) in enumerate(zip(actual, expected)):
            result = self._check_approx_num(a, e, tol, rel, msg, i)
            if result is not None:
                return result

    def _check_approx_num(self, actual, expected, tol, rel, msg, idx=None):
        # Note that we reverse the order of the arguments.
        if approx_equal(expected, actual, tol, rel):
            # Test passes. Return early, we are done.
            return None
        # Otherwise we failed. Generate an exception and return it.
        standardMsg = self._make_std_err_msg(actual, expected, tol, rel, idx)
        msg = self._formatMessage(msg, standardMsg)
        return self.failureException(msg)  # Don't raise.

    def _make_std_err_msg(self, actual, expected, tol, rel, idx):
        # Create the standard error message, starting with the common part,
        # which comes at the end.
        abs_err = abs(actual - expected)
        rel_err = abs_err/abs(expected) if expected else float('inf')
        err_msg = '    absolute error = %r\n    relative error = %r'
        # Now for the part that is not common to all messages.
        if idx is None:
            # Comparing two numeric values.
            idxheader = ''
        else:
            idxheader = 'numeric sequences first differs at index %d.\n' % idx
        if tol is rel is None:
            header = 'actual value %r is not equal to expected %r\n'
            items = (actual, expected, abs_err, rel_err)
        else:
            header = 'actual value %r differs from expected %r\n' \
                        '    by more than %s\n'
            t = []
            if tol is not None:
                t.append('tol=%r' % tol)
            if rel is not None:
                t.append('rel=%r' % rel)
            assert t
            items = (actual, expected, ' and '.join(t), abs_err, rel_err)
        standardMsg = (idxheader + header + err_msg) % items
        return standardMsg

