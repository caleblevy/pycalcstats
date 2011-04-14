#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##  Copyright (c) 2011 Steven D'Aprano.
##  See the file stats/__init__.py for the licence terms for this software.

"""Test code for the top-level module of the stats package."""

# FIXME Tests with random data currently cannot be replicated if they fail.
# Such failed tests should be recorded for retesting. (See also fuzz testing?)


import collections
import functools
import itertools
import math
import operator
import random
import unittest

from decimal import Decimal
from fractions import Fraction

# The module(s) to be tested:
import stats


# === Helper functions ===

def approx_equal(x, y, tol=1e-12, rel=1e-7):
    """Test whether x is approximately equal to y, using an absolute error
    of tol and/or a relative error of rel, whichever is bigger.

    Pass None as either tol or rel to ignore that test; if both are None,
    the test performed is an exact equality test.

    tol and rel must be either None or a positive, finite number, otherwise
    the behaviour is undefined.
    """
    # Note that the relative error is calculated relative to x only.
    if tol is rel is None:
        # Fall back on exact equality.
        return x == y
    # Infinities and NANs are special.
    if math.isnan(x) or math.isnan(y):
        return False
    delta = abs(x - y)
    if math.isnan(delta):
        # Only if both x and y are the same infinity.
        assert x == y and math.isinf(x)
        return True
    if math.isinf(delta):
        # Either x and y are both infinities with the opposite sign, or
        # one is an infinity and the other is finite. Either way, they're
        # not approximately equal.
        return False
    tests = []
    if tol is not None: tests.append(tol)
    if rel is not None: tests.append(rel*abs(x))
    assert tests
    return delta <= max(tests)


# === Unit tests ===

# Note: do not use self.fail... unit tests, as they are deprecated in
# Python 3.2. Although plural test cases such as self.testEquals and
# friends are not officially deprecated, they are discouraged.


# -- Generic test suite subclass --
# We prefer this for testing numeric values that may not be exactly equal.
# Avoid using TestCase.almost_equal, because it sucks :)

USE_DEFAULT = object()
class NumericTestCase(unittest.TestCase):
    # By default, we expect exact equality.
    tol = None
    rel = None

    def assertApproxEqual(
        self, actual, expected, tol=USE_DEFAULT, rel=USE_DEFAULT, msg=None
        ):
        if tol is USE_DEFAULT: tol = self.tol
        if rel is USE_DEFAULT: rel = self.rel
        if (isinstance(actual, collections.Sequence) and
        isinstance(expected, collections.Sequence)):
            checker = self._check_approx_seq
        else:
            checker = self._check_approx_num
        result = checker(actual, expected, tol, rel, msg)
        if result:
            raise result

    def _check_approx_seq(self, actual, expected, tol, rel, msg):
        if len(actual) != len(expected):
            standardMsg = (
                "actual and expected sequences differ in length; expected"
                " %d items but found %d." % (len(expected), len(actual)))
            msg = self._formatMessage(msg, standardMsg)
            # DON'T raise the exception, return it to be raised later!
            return self.failureException(msg)
        for i, (a,e) in enumerate(zip(actual, expected)):
            result = self._check_approx_num(a, e, tol, rel, msg, i)
            if result is not None:
                return result

    def _check_approx_num(self, actual, expected, tol, rel, msg, idx=None):
        # Note that we reverse the order of the arguments here:
        if approx_equal(expected, actual, tol, rel):
            # Test passes. Return early, we are done.
            return None
        # Otherwise we failed. Generate an exception and return it.
        standardMsg = self._make_std_err_msg(actual, expected, tol, rel, idx)
        msg = self._formatMessage(msg, standardMsg)
        # DON'T raise the exception, return it to be raised later!
        return self.failureException(msg)

    @staticmethod
    def _make_std_err_msg(actual, expected, tol, rel, idx):
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


# -- Test the test infrastructure ---

class ApproxTest(unittest.TestCase):
    # Test the approx_equal test helper function.

    def getvalues(self):
        """Return a random set of values."""
        values = [random.gauss(0, 1) for _ in range(50)]
        values.extend([random.gauss(0, 10000) for _ in range(50)])
        return values

    def _equality_tests(self, x, y):
        """Test approx_equal various ways."""
        return (approx_equal(x, y),
                approx_equal(x, y, tol=None),
                approx_equal(x, y, rel=None),
                approx_equal(x, y, tol=None, rel=None),
                approx_equal(y, x),
                approx_equal(y, x, tol=None),
                approx_equal(y, x, rel=None),
                approx_equal(y, x, tol=None, rel=None),
                )

    def testEqualFixed(self):
        # Test equality with a fixed set of values.
        values = [-42, -23, -2, -1, 0, 1, 2, 3, 17, 35,
                  -1e145, -2e63, 4e58, 7e123,
                  -123.456, -1.1, 0.5, 1.9, 23.42, 0.0]
        values.append(math.copysign(0, -1))
        for x in values:
            # x should always equal x, for any finite x.
            results = self._equality_tests(x, x)
            self.assertTrue(all(results), 'equality failure for x=%r' % x)

    def testEqualRandom(self):
        # Test equality with a random set of values.
        values = self.getvalues()
        for x in values:
            results = self._equality_tests(x, x)
            self.assertTrue(all(results), 'equality failure for x=%r' % x)

    def testUnequal(self):
        values1 = self.getvalues()
        values2 = self.getvalues()
        for x, y in zip(values1, values2):
            if x == y: # Not very likely, but it could happen.
                continue
            results = self._equality_tests(x, y)
            self.assertFalse(any(results),
                             'inequality failure for x=%r, y=%r' % (x,y))

    def testSpecials(self):
        # Test approx_equal behaviour with infinities and NANs.
        inf = float('inf')
        ninf = float('-inf')
        nan = float('nan')
        # +inf == +inf regardless of precision
        # -inf == -inf regardless of precision
        for x in (inf, ninf):
            results = self._equality_tests(x, x)
            self.assertTrue(all(results), 'equality failure for x=%r' % x)
        # +inf != -inf regardless of precision
        results = self._equality_tests(inf, ninf)
        self.assertFalse(any(results),
            'expected +inf != -inf but found equal')
        # inf != finite regardless of precision
        for x in (inf, ninf):
            for y in (0.0, 1.0, 3.5, 7e125):
                results = self._equality_tests(x, y)
                self.assertFalse(any(results),
                    'inequality failure for x=%r, y=%r' % (x,y))
        # nan != anything regardless of precision
        for x in (inf, ninf, nan, 0.0, 1.0, 3.5, 7e125):
            results = self._equality_tests(nan, x)
            self.assertFalse(any(results),
                'nan inequality failure for x=%r' % x)

    def testAbsolute(self):
        x = random.uniform(-23, 42)
        for tol in (1e-13, 1e-12, 1e-10, 1e-5):
            # Test delta < tol.
            self.assertTrue(approx_equal(x, x+tol/2, tol, None))
            self.assertTrue(approx_equal(x, x-tol/2, tol, None))
            # Test delta > tol.
            self.assertFalse(approx_equal(x, x+tol*2, tol, None))
            self.assertFalse(approx_equal(x, x-tol*2, tol, None))
            # With delta == tol exactly, rounding errors can make
            # the test fail.

    def testRelative(self):
        for x in (1e-10, 1.1, 123.456, 1.23456e18, -17.98):
            for rel in (1e-2, 1e-4, 1e-7, 1e-9):
                # Test delta < rel.
                delta = x*rel/2
                self.assertTrue(approx_equal(x, x+delta, None, rel))
                self.assertTrue(approx_equal(x, x+delta, None, rel))
                # Test delta > rel.
                delta = x*rel*2
                self.assertFalse(approx_equal(x, x+delta, None, rel))
                self.assertFalse(approx_equal(x, x+delta, None, rel))
                # With delta == rel exactly, rounding errors can make
                # the test fail.


class TestNumericTestCase(unittest.TestCase):
    # The formatting routine that generates the error messages is complex
    # enough that it needs its own test.

    def test_error_msg_exact(self):
        # Test the error message generated for exact tests.
        msg = NumericTestCase._make_std_err_msg(0.5, 0.25, None, None, None)
        self.assertEqual(msg,
            "actual value 0.5 is not equal to expected 0.25\n"
            "    absolute error = 0.25\n"
            "    relative error = 1.0"
            )

    def test_error_msg_inexact(self):
        # Test the error message generated for inexact tests.
        msg = NumericTestCase._make_std_err_msg(2.25, 1.25, 0.25, None, None)
        self.assertEqual(msg,
            "actual value 2.25 differs from expected 1.25\n"
            "    by more than tol=0.25\n"
            "    absolute error = 1.0\n"
            "    relative error = 0.8"
            )
        msg = NumericTestCase._make_std_err_msg(1.5, 2.5, None, 0.25, None)
        self.assertEqual(msg,
            "actual value 1.5 differs from expected 2.5\n"
            "    by more than rel=0.25\n"
            "    absolute error = 1.0\n"
            "    relative error = 0.4"
            )
        msg = NumericTestCase._make_std_err_msg(2.5, 4.0, 0.5, 0.25, None)
        self.assertEqual(msg,
            "actual value 2.5 differs from expected 4.0\n"
            "    by more than tol=0.5 and rel=0.25\n"
            "    absolute error = 1.5\n"
            "    relative error = 0.375"
            )

    def test_error_msg_sequence(self):
        # Test the error message generated for sequence tests.
        msg = NumericTestCase._make_std_err_msg(2.5, 4.0, 0.5, 0.25, 7)
        self.assertEqual(msg,
            "numeric sequences first differs at index 7.\n"
            "actual value 2.5 differs from expected 4.0\n"
            "    by more than tol=0.5 and rel=0.25\n"
            "    absolute error = 1.5\n"
            "    relative error = 0.375"
            )


# -- Test mixins --

class MetadataMixin:
    expected_metadata = ["__doc__", "__all__"]

    def testMeta(self):
        # Test for the existence of metadata.
        for meta in self.expected_metadata:
            self.assertTrue(hasattr(self.module, meta),
                            "%s not present" % meta)


class TestConsumerMixin:
    def testIsConsumer(self):
        # Test that the function is a consumer.
        cr = self.func()
        self.assertTrue(hasattr(cr, 'send'))


class UnivariateMixin:
    # Common tests for most univariate functions that take a data argument.
    #
    # This tests the behaviour of functions of the form func(data [,...])
    # without checking the value returned. Tests for correctness of the
    # return value are *not* the responsibility of this class.


    def testNoArgs(self):
        # Fail if given no arguments.
        self.assertRaises(TypeError, self.func)

    def testEmptyData(self):
        # Fail when the data argument (first argument) is empty.
        for empty in ([], (), iter([])):
            self.assertRaises(ValueError, self.func, empty)

    def testSingleData(self):
        # Pass when the first argument has a single data point.
        for x in self.make_random_data(size=1, count=4):
            _ = self.func([x])

    def testDoubleData(self):
        # Pass when the first argument has two data points.
        for x,y in self.make_random_data(size=2, count=4):
            _ = self.func([x,y])

    def testTripleData(self):
        # Pass when the first argument has three data points.
        for x,y,z in self.make_random_data(size=3, count=4):
            _ = self.func([x,y,z])

    # Most stats functions won't care much about the length of the input
    # data, provided there are sufficient data points (usually >= 1). But
    # when testing the functions (particularly those in stats.order), we
    # MUST care about the length: we need to cover each case where the
    # data has a multiple of 4 items, plus 0-3 remainders (that is, where
    # len(data)%4 = 0, 1, 2, 3). This ensures that all four internal code
    # paths are tested.

    def testQuadPlusData(self):
        # Pass when the first argument has four + data points.
        for n in range(4, 12):
            for t in self.make_random_data(size=n, count=3):
                _ = self.func(t)

    def make_random_data(self, size, count):
        """Return count lists of random data, each of given size."""
        data = []
        for i in range(count):
            data.append([random.random() for j in range(size)])
        assert len(data) == count
        assert all(len(t) == size for t in data)
        return data

    def testInPlaceModifications(self):
        # Test that the function does not modify its input data.
        for n in range(4, 12):
            datasets = self.make_random_data(size=n, count=3)
            for data in datasets:
                # Make sure that the data isn't sorted, because some
                # functions being tested may sort the data. If we don't
                # shuffle the data, the test will fail purely by accident.
                sorted_data = sorted(data)
                assert len(data) != 1  # Avoid infinite loops.
                while data == sorted_data:
                    random.shuffle(data)
                # Now we know that data is not in sorted order. If the
                # function being tested sorts it in place, we can detect
                # the change.
                assert data != sorted_data
                saved_data = data[:]
                assert data is not saved_data
                _ = self.func(data)
                self.assertEqual(data, saved_data, "data has been modified")

    def testOrderOfDataPoints(self):
        # Test that the result of the function shouldn't depend on the
        # order of data points. In practice, due to floating point
        # rounding, it may depend slightly.
        for n in range(4, 12):
            datasets = self.make_random_data(size=n, count=3)
            for data in datasets:
                data.sort()
                expected = self.func(data)
                result = self.func(reversed(data))
                self.assertApproxEqual(expected, result)
                for i in range(10):
                    random.shuffle(data)
                    result = self.func(data)
                    self.assertApproxEqual(result, expected)

    def testTypeOfDataCollection(self):
        # Test that the type of iterable data doesn't effect the result.
        class MyList(list):
            pass
        class MyTuple(tuple):
            pass
        def generator(data):
            return (obj for obj in data)

        for n in range(4, 12):
            # Start with a range object as data.
            data = range(n)
            expected = self.func(data)
            for kind in (list, tuple, iter, MyList, MyTuple, generator):
                result = self.func(kind(data))
                self.assertEqual(result, expected)

    def testTypeOfDataElement(self):
        # Test that the type of data elements shouldn't effect the result.
        class MyFloat(float):
            pass

        for n in range(4, 12):
            datasets = self.make_random_data(size=n, count=3)
            for data in datasets:
                expected = self.func(data)
                data = [MyFloat(x) for x in data]
                result = self.func(data)
                self.assertEqual(result, expected)

    def testBadArgType(self):
        # Test failures with bad argument types.
        for d in (None, 23, object(), "spam"):
            self.assertRaises(TypeError, self.func, d)


# -- Tests for the stats module --

class GlobalsTest(unittest.TestCase, MetadataMixin):
    module = stats

    def testCheckAll(self):
        # Check everything in __all__ exists.
        module = self.module
        for name in module.__all__:
            # No private names in __all__:
            self.assertFalse(name.startswith("_"))
            # And anything in __all__ must exist:
            self.assertTrue(hasattr(module, name))


class ExtraMetadataTest(unittest.TestCase, MetadataMixin):
    expected_metadata = [
            "__version__", "__date__", "__author__", "__author_email__"
            ]
    module = stats


class StatsErrorTest(unittest.TestCase):
    def testHasException(self):
        self.assertTrue(hasattr(stats, 'StatsError'))
        self.assertTrue(issubclass(stats.StatsError, ValueError))


class AddPartialTest(unittest.TestCase):
    def testInplace(self):
        # Test that add_partial modifies list in place and returns None.
        L = []
        result = stats.add_partial(1.5, L)
        self.assertEqual(L, [1.5])
        self.assertTrue(result is None)

    def testAdd(self):
        # Test that add_partial actually does add.
        L = []
        stats.add_partial(1.5, L)
        stats.add_partial(2.5, L)
        self.assertEqual(sum(L), 4.0)
        stats.add_partial(1e120, L)
        stats.add_partial(1e-120, L)
        stats.add_partial(0.5, L)
        self.assertEqual(sum(L), 1e120)
        stats.add_partial(-1e120, L)
        self.assertEqual(sum(L), 4.5)
        stats.add_partial(-4.5, L)
        self.assertEqual(sum(L), 1e-120)


class SumTest(NumericTestCase, UnivariateMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.sum

    def testEmptyData(self):
        # Override method from UnivariateMixin.
        for empty in ([], (), iter([])):
            self.assertEqual(self.func(empty), 0)
            self.assertEqual(self.func(empty, 123.456), 123.456)
            self.assertEqual(self.func(empty, [1,2,3]), [1,2,3])

    def testFloatSum(self):
        # Compare with the math.fsum function.
        data = [random.uniform(-100, 1000) for _ in range(1000)]
        self.assertEqual(self.func(data), math.fsum(data))

    def testColumnsSum(self):
        # Test adding up columns.
        columns = [[1, 2, 3, 4, 5, 6, 7, 8, 9],
                   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                   [0, -1, -3, -2, -5, -4, 1, 3, 2],
                   [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
                   [0.1, 1e100, 0.1, -1e100, 0.1, 1e100, 0.1, -1e100, 0.1],
                   [1e80, 1e70, 1e75, 1e80, 1e79, 1e76, 1e80, 1e77, 1e78],
                   ]
        assert all(len(col) == 9 for col in columns)
        expected = [math.fsum(col) for col in columns]
        self.assertEqual(self.func(zip(*columns)), expected)

    def testColumnErrors(self):
        # Test that rows must have the same number of columns.
        data = [[1, 2, 3], [1, 2]]
        self.assertRaises(ValueError, self.func, data)

    def testExactSeries(self):
        # Compare with exact formulae for certain mathematical series.
        # sum of 1, 2, 3, ... n = n(n+1)/2
        data = range(1, 131)
        expected = 130*131/2
        self.assertEqual(self.func(data), expected)
        # sum of squares of 1, 2, 3, ... n = n(n+1)(2n+1)/6
        data = [n**2 for n in range(1, 57)]
        expected = 56*57*(2*56+1)/6
        self.assertEqual(self.func(data), expected)
        # sum of cubes of 1, 2, 3, ... n = n**2(n+1)**2/4 = (1+2+...+n)**2
        data1 = range(1, 85)
        data2 = [n**3 for n in data1]
        expected = (84**2*85**2)/4
        self.assertEqual(self.func(data1)**2, expected)
        self.assertEqual(self.func(data2), expected)

    def testStartArgument(self):
        # Test that the optional start argument works correctly.
        data = [random.uniform(1, 1000) for _ in range(100)]
        t = self.func(data)
        self.assertEqual(t+42, self.func(data, 42))
        self.assertEqual(t-23, self.func(data, -23))
        self.assertEqual(t+1e20, self.func(data, 1e20))

    def testStartArgumentVectors(self):
        # Test that the optional start argument works correctly for vectors.
        data = []
        for i in range(10):
            # Add a row.
            data.append([random.uniform(1, 1000) for _ in range(100)])
        columns = self.func(data)
        # Test with a scalar start value.
        for start in (42, -23, 1e20):
            expected = [x+start for x in columns]
            self.assertEqual(self.func(data, start), expected)
        # Test with a vector start value.
        start = [random.uniform(1, 1000) for _ in range(100)]
        assert len(start) == len(columns)
        expected = [a+b for a,b in zip(columns, start)]
        self.assertEqual(self.func(data, start), expected)

    def testStartArgumentErrors(self):
        # Test optional start argument failure modes.
        data = [[1, 2, 3], [4, 5, 6]]
        self.assertRaises(ValueError, self.func, data, [1, 2])
        self.assertRaises(ValueError, self.func, data, [1, 2, 3, 4])
        data = [1, 2, 3, 4]
        self.assertRaises(TypeError, self.func, data, [1, 2, 3, 4])


class SumTortureTest(NumericTestCase):
    def testTorture(self):
        # Tim Peters' torture test for sum, and variants of same.
        func = stats.sum
        self.assertEqual(func([1, 1e100, 1, -1e100]*10000), 20000.0)
        self.assertEqual(func([1e100, 1, 1, -1e100]*10000), 20000.0)
        self.assertApproxEqual(
            func([1e-100, 1, 1e-100, -1]*10000), 2.0e-96, tol=1e-15)
            # *raises eyebrow* -- result = 2e-96, +/- 1e-15 ???


class RunningSumTest(unittest.TestCase, TestConsumerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.running_sum

    def testSum(self):
        cr = self.func()
        self.assertEqual(cr.send(3), 3)
        self.assertEqual(cr.send(5), 8)
        self.assertEqual(cr.send(0), 8)
        self.assertEqual(cr.send(-2), 6)
        self.assertEqual(cr.send(0.5), 6.5)
        self.assertEqual(cr.send(2.75), 9.25)

    def testSumStart(self):
        cr = self.func(12)
        self.assertEqual(cr.send(3), 15)
        self.assertEqual(cr.send(5), 20)
        self.assertEqual(cr.send(0), 20)
        self.assertEqual(cr.send(-2), 18)
        self.assertEqual(cr.send(0.5), 18.5)
        self.assertEqual(cr.send(2.75), 21.25)

    def testSumTortureTest(self):
        cr = self.func()
        for i in range(100):
            self.assertEqual(cr.send(1), 2*i+1)
            self.assertEqual(cr.send(1e100), 1e100)
            self.assertEqual(cr.send(1), 1e100)
            self.assertEqual(cr.send(-1e100), 2*i+2)


class ProductTest(NumericTestCase, UnivariateMixin):
    rel = 1e-14

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.product

    def testEmptyData(self):
        # Override method from UnivariateMixin.
        for empty in ([], (), iter([])):
            self.assertEqual(self.func(empty), 1)
            self.assertEqual(self.func(empty, 21.3), 21.3)
            self.assertEqual(self.func(empty, [1,2,3]), [1,2,3])

    def testTorture(self):
        # Torture test for product.
        data = []
        for i in range(1, 101):
            data.append(i)
            data.append(1/i)
        self.assertApproxEqual(self.func(data), 1.0, tol=1e-14)
        random.shuffle(data)
        self.assertApproxEqual(self.func(data), 1.0, tol=1e-14)

    def testProduct(self):
        self.assertEqual(self.func([1.5, 5.0, 7.5, 12.0]), 675.0)
        self.assertEqual(self.func([5]*20), 5**20)
        data = [i/(i+1) for i in range(1, 1024)]
        random.shuffle(data)
        self.assertApproxEqual(self.func(data), 1/1024)
        self.assertApproxEqual(self.func(data, 2.5), 5/2048)

    def testNegatives(self):
        self.assertEqual(self.func([2, -3]), -6)
        self.assertEqual(self.func([2, -3, -4]), 24)
        self.assertEqual(self.func([-2, 3, -4, -5]), -120)

    def testFact(self):
        self.assertEqual(self.func(range(1, 7)), math.factorial(6))
        self.assertEqual(self.func(range(1, 24)), math.factorial(23))

    def testColumns(self):
        # Test multiplying columns.
        columns = [[1, 2, 3, 4, 5, 6, 7, 8, 9],
                   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                   [1e20, 3e20, 4e21, 1e19, 3e16, 2e17, 1e18, 9e17, 5e18],
                   [-4, 1, 3, 2, 0, -1, -3, -2, -5],
                   [0.7, 1.3, 2.9, 3.3, 5.6, 6.0, 7.1, 8.3, 9.4],
                   ]
        assert all(len(col) == 9 for col in columns)
        expected = [functools.reduce(operator.mul, col) for col in columns]
        self.assertEqual(self.func(zip(*columns)), expected)

    def testColumnErrors(self):
        # Test that rows must have the same number of columns.
        data = [[1, 2, 3], [1, 2]]
        self.assertRaises(ValueError, self.func, data)

    def testStartArgument(self):
        # Test that the optional start argument works correctly.
        data = [random.uniform(1, 50) for _ in range(30)]
        t = self.func(data)
        for start in (42, 0.2, -23):
            self.assertApproxEqual(t*start, self.func(data, start))
        self.assertApproxEqual(t*1e120, self.func(data, 1e120))

    def testStartArgumentVectors(self):
        # Test that the optional start argument works correctly for vectors.
        data = []
        for i in range(10):
            # Add a row.
            data.append([random.uniform(2, 12) for _ in range(100)])
        columns = self.func(data)
        # Test with a scalar start value.
        for start in (2.5, 17.1, 123):
            expected = [x*start for x in columns]
            self.assertApproxEqual(self.func(data, start), expected)
        # Test with a vector start value.
        start = [random.uniform(1, 10) for _ in range(100)]
        assert len(start) == len(columns)
        expected = [a*b for a,b in zip(columns, start)]
        self.assertApproxEqual(self.func(data, start), expected)

    def testStartArgumentErrors(self):
        # Test optional start argument failure modes.
        data = [[1, 2, 3], [4, 5, 6]]
        self.assertRaises(ValueError, self.func, data, [1, 2])
        self.assertRaises(ValueError, self.func, data, [1, 2, 3, 4])
        data = [1, 2, 3, 4]
        self.assertRaises(TypeError, self.func, data, [1, 2, 3, 4])

    def testZero(self):
        # Product of anything containing zero is always zero.
        for data in (range(23), range(-35, 36)):
            self.assertEqual(self.func(data), 0)


class MeanTest(NumericTestCase, UnivariateMixin):
    # We expect this to be subclassed by tests for the other means.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.mean
        self.data = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
        self.expected = 5.5

    def setUp(self):
        random.shuffle(self.data)

    def testSeq(self):
        self.assertApproxEqual(self.func(self.data), self.expected)

    def testBigData(self):
        data = [x + 1e9 for x in self.data]
        expected = self.expected + 1e9
        assert expected != 1e9
        self.assertApproxEqual(self.func(data), expected)

    def testIter(self):
        self.assertApproxEqual(self.func(iter(self.data)), self.expected)

    def testSingleton(self):
        for x in self.data:
            self.assertEqual(self.func([x]), x)

    def testDoubling(self):
        # Average of [a,b,c...z] should be same as for [a,a,b,b,c,c...z,z].
        data = [random.random() for _ in range(1000)]
        a = self.func(data)
        b = self.func(data*2)
        self.assertApproxEqual(a, b)

    def testColumns(self):
        # Test columnar data.
        columns = [[1, 2, 3, 4, 5, 6, 7, 8, 9],
                   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                   [0, -1, -3, -2, -5, -4, 1, 3, 2],
                   [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
                   [0.1, 1e100, 0.1, -1e100, 0.1, 1e100, 0.1, -1e100, 0.1],
                   [1e80, 1e70, 1e75, 1e80, 1e79, 1e76, 1e80, 1e77, 1e78],
                   ]
        assert all(len(col) == 9 for col in columns)
        expected = [self.func(col) for col in columns]
        self.assertEqual(self.func(zip(*columns)), expected)

    def testColumnErrors(self):
        # Test that rows must have the same number of columns.
        data = [[1, 2, 3], [1, 2]]
        self.assertRaises(ValueError, self.func, data)


class ExactVarianceTest(NumericTestCase):
    # Exact tests for variance and friends.
    tol = 5e-11
    rel = 5e-16

    def testVariance(self):
        data = [1, 2, 3]
        assert stats.mean(data) == 2
        self.assertEqual(stats.pvariance(data), 2/3)
        self.assertEqual(stats.variance(data), 1.0)
        self.assertEqual(stats.pstdev(data), math.sqrt(2/3))
        self.assertEqual(stats.stdev(data), 1.0)

    def testKnownUnbiased(self):
        # Test that variance is unbiased with known data.
        data = [1, 1, 2, 5]  # Don't give data too many items!
        samples = self.get_all_samples(data)
        assert stats.mean(data) == 2.25
        assert stats.pvariance(data) == 2.6875
        sample_variances = [stats.variance(sample) for sample in samples]
        self.assertEqual(stats.mean(sample_variances), 2.6875)

    def testRandomUnbiased(self):
        # Test that variance is unbiased with random data.
        data = [random.uniform(-100, 1000) for _ in range(5)]
        samples = self.get_all_samples(data)
        pvar = stats.pvariance(data)
        sample_variances = [stats.variance(sample) for sample in samples]
        self.assertApproxEqual(stats.mean(sample_variances), pvar)

    def get_all_samples(self, data):
        """Return a generator that returns all permutations with
        replacement of the given data."""
        return itertools.chain(
            *(itertools.product(data, repeat=n) for n in
            range(2, len(data)+1)))


class PVarianceTest(NumericTestCase, UnivariateMixin):
    # Test population variance.
    # This will be subclassed by variance and [p]stdev.

    tol = 1e-11

    def __init__(self, *args, **kwargs):
        NumericTestCase.__init__(self, *args, **kwargs)
        self.func = stats.pvariance
        # Test data for test_main, test_shift:
        self.data = [4.0, 7.0, 13.0, 16.0]
        self.expected = 22.5  # Exact population variance of self.data.
        # Test data for test_uniform:
        self.uniform_data = range(10000)
        self.uniform_expected = (10000**2 - 1)/12  # Exact value.
        # If you duplicate each data point, the variance will scale by
        # this value:
        self.duplication_scale_factor = 1.0
        # Expected result calculated by HP-48GX -- see testCompareHP.
        self.hp_expected = 88349.2408884

    def setUp(self):
        random.shuffle(self.data)

    def test_main(self):
        # Test that pvariance calculates the correct result.
        self.assertEqual(self.func(self.data), self.expected)

    def test_shift(self):
        # Shifting the data by a constant amount should not affect
        # the variance.
        for shift in (1e2, 1e6, 1e9):
            data = [x + shift for x in self.data]
            self.assertEqual(self.func(data), self.expected)

    def test_uniform(self):
        # Compare the calculated variance against an exact result.
        data = list(self.uniform_data)
        random.shuffle(data)
        self.assertEqual(self.func(data), self.uniform_expected)

    def test_equal_data(self):
        # If the data is constant, the variance should be zero.
        self.assertEqual(self.func([42]*10), 0)

    def testDuplicate(self):
        # Test that the variance behaves as expected when you duplicate
        # each data point [a,b,c,...] -> [a,a,b,b,c,c,...]
        data = [random.uniform(-100, 500) for _ in range(20)]
        expected = self.func(data)*self.duplication_scale_factor
        actual = self.func(data*2)
        self.assertApproxEqual(actual, expected)

    def testCompareHP(self):
        # Compare against a result calculated with a HP-48GX calculator.
        data = (list(range(1, 11)) + list(range(1000, 1201)) +
            [0, 3, 7, 23, 42, 101, 111, 500, 567])
        random.shuffle(data)
        self.assertApproxEqual(self.func(data), self.hp_expected, tol=1e-7)

    def testDomainError(self):
        # Domain error exception reported by Geremy Condra.
        data = [0.123456789012345]*10000
        # All the items are identical, so variance should be exactly zero.
        # We allow some small round-off error.
        self.assertApproxEqual(self.func(data), 0.0, tol=5e-17)

    def testColumns(self):
        # Test columnar data.
        columns = [[1, 2, 3, 4, 5, 6, 7, 8, 9],
                   [5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9],
                   [-0.2, 0.3, 0.5, 0.7, 1.2, 1.2, 1.2, 1.5, 3.4],
                   [1e80, 3e76, 2e75, 1e80, 9e79, 7e76, 2e80, 5e77, 9e78],
                   [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9],
                   [3, -7, 2, -5, 0, 1, 4, -6, 5],
                   [0.2, 1.7, 2.3, 4.2, 5.0, 6.1, 7.3, 8.8, 9.6],
                   ]
        assert all(len(col) == 9 for col in columns)
        expected = [self.func(col) for col in columns]
        self.assertApproxEqual(self.func(zip(*columns)), expected, tol=1e-14)

    def testColumnErrors(self):
        # Test that rows must have the same number of columns.
        data = [[1, 2, 3], [1, 2]]
        self.assertRaises(ValueError, self.func, data)

    def testSingleton(self):
        # Population variance of a single value is always zero.
        for x in self.data:
            self.assertEqual(self.func([x]), 0)

    def testMeanArgument(self):
        # Variance calculated with the given mean should be the same
        # as that calculated without the mean.
        data = [random.random() for _ in range(15)]
        m = stats.mean(data)
        expected = self.func(data)
        self.assertEqual(self.func(data, m), expected)

    def testVectorMeanArgument(self):
        N = 15
        columns = [[random.random() for _ in range(N)],
                   [random.uniform(-100, 100) for _ in range(N)],
                   [random.uniform(1, 1000) for _ in range(N)],
                   [random.gauss(3, 2) for _ in range(N)],
                   ]
        assert all(len(col) == N for col in columns)
        means = [stats.mean(col) for col in columns]
        expected = self.func(zip(*columns))
        actual = self.func(zip(*columns), means)
        self.assertApproxEqual(actual, expected, tol=1e-14)

    def testMeanArgumentErrors(self):
        # Test optional mean argument failure modes.
        data = [[1, 2, 3], [4, 5, 6]]
        self.assertRaises(ValueError, self.func, data, [1, 2])
        self.assertRaises(ValueError, self.func, data, [1, 2, 3, 4])
        data = [1, 2, 3, 4]
        self.assertRaises(TypeError, self.func, data, [1, 2, 3, 4])


class PVarianceDupsTest(NumericTestCase):
    tol=1e-12

    def testManyDuplicates(self):
        from stats import pvariance
        # Start with 1000 normally distributed data points.
        data = [random.gauss(7.5, 5.5) for _ in range(1000)]
        expected = pvariance(data)
        # We expect the calculated variance to be close to the exact result
        # for the variance, namely 5.5**2, but because the data was
        # generated randomly, it might not be. Either way, it doesn't matter.
        #
        # Duplicating the data points should keep the variance the same.
        for n in (3, 5, 10, 20, 30):
            d = data*n
            actual = pvariance(d)
            self.assertApproxEqual(actual, expected)
        # Now try again with a *lot* of duplicates.
        def big_data():
            for _ in range(500):
                for x in data:
                    yield x
        actual = pvariance(big_data())
        self.assertApproxEqual(actual, expected)


class VarianceTest(PVarianceTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.variance
        self.expected = 30.0  # Exact sample variance of self.data.
        self.uniform_expected = self.uniform_expected * 10000/(10000-1)
        self.hp_expected = 88752.6620797
        # Scaling factor when you duplicate each data point:
        self.duplication_scale_factor = (2*20-2)/(2*20-1)

    def testCompareR(self):
        # Compare against a result calculated with R code:
        #   > x <- c(seq(1, 10), seq(1000, 1200))
        #   > var(x)
        #   [1] 57563.55
        data = list(range(1, 11)) + list(range(1000, 1201))
        expected = 57563.550
        self.assertApproxEqual(self.func(data), expected, tol=1e-3)
        # The expected value from R looks awfully precise... are they
        # rounding it, or is that the exact value?
        # My HP-48GX calculator returns 57563.5502144.

    def testSingleData(self):
        # Override mixin test.
        self.assertRaises(stats.StatsError, self.func, [23])

    def testSingleton(self):
        # Override pvariance test.
        self.assertRaises(stats.StatsError, self.func, [42])
        # Three columns, each with one data point.
        self.assertRaises(stats.StatsError, self.func, [[23, 42, 99]])


class PStdevTest(PVarianceTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.pstdev
        self.expected = math.sqrt(self.expected)
        self.uniform_expected = math.sqrt(self.uniform_expected)
        self.hp_expected = 297.236002006
        self.duplication_scale_factor = math.sqrt(self.duplication_scale_factor)


class StdevTest(VarianceTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.stdev
        self.expected = math.sqrt(self.expected)
        self.uniform_expected = math.sqrt(self.uniform_expected)
        self.hp_expected = 297.913850097
        self.duplication_scale_factor = math.sqrt(self.duplication_scale_factor)

    def testCompareR(self):
        data = list(range(1, 11)) + list(range(1000, 1201))
        expected = 239.9241
        self.assertApproxEqual(self.func(data), expected, tol=1e-4)


class VarianceMeanScalarTest(NumericTestCase):
    # Additional variance calculations when the mean is explicitly supplied.

    def make_data(self):
        mu = 100*random.random()
        sigma = 10*random.random()+1
        return (
            [-6, -2, 0, 3, 4, 5, 5, 5, 6, 7, 9, 11, 15, 25, 26, 27, 28, 42],
            [random.random() for _ in range(10)],
            [random.uniform(10000, 11000) for _ in range(50)],
            [random.gauss(mu, sigma) for _ in range(50)],
            )

    def compare_with_and_without_mean(self, func):
        for data in self.make_data():
            m = stats.mean(data)
            expected = func(data)
            actual = func(data, m)
            self.assertEqual(actual, expected)

    def test_pvar(self):
        self.compare_with_and_without_mean(stats.pvariance)

    def test_var(self):
        self.compare_with_and_without_mean(stats.variance)

    def test_pstdev(self):
        self.compare_with_and_without_mean(stats.pstdev)

    def test_stdev(self):
        self.compare_with_and_without_mean(stats.stdev)


class VarianceMeanVectorTest(VarianceMeanScalarTest):
    # Test variance and friends with vectorized data.

    def make_data(self):
        return ([ [7.3409, 3.1025, 12, -3, -9.36, 2.6731],
                  [6.0815, 2.4881, 15, +1, -7.84, 3.1852],
                  ],
                [ [2, 4], [9, 8], [1, 3], [3, 4], [5, 7], [2, 6],
                  [0, 0], [1, 0], [5, 1], [7, 9], [3, 7], [8, 5],
                  ],
                [ [3.25e7, 5.93e-9, 2.04e10, 1.87e12],
                  [5.01e7, 2.18e-8, 7.32e11, 1.44e12],
                  [2.46e7, 4.29e-8, 9.99e11, 1.92e12],
                  ],
                [ [3, 5, 7, 2, 6, 2, 3, 9, 1, 4, 2, 0, 4, 9, 3, 4, 8, 1],
                  [1, 7, 0, 8, 0, 5, 1, 3, 9, 4, 0, 5, 9, 1, 3, 7, 2, 6],
                  [5, 1, 2, 0, 1, 8, 0, 5, 9, 9, 0, 8, 4, 1, 7, 1, 2, 3],
                  [2, 1, 0, 5, 9, 2, 6, 7, 3, 9, 0, 8, 2, 5, 1, 7, 1, 0],
                  ],
                [ [1.0, 3.4, 5.1, 7.9, 9.8],
                  [0.1, 2.6, 4.0, 6.3, 8.5],
                  [2.5, 3.3, 4.5, 5.7, 6.0],
                  [0.5, 1.5, 5.0, 3.1, 2.4],
                  [3.0, 5.8, 9.1, 1.1, 2.8],
                  [1.3, 3.0, 7.4, 9.1, 8.7],
                  ],
                [ [-3, 4, 5, 6, 7, 0, 1, 2, 8, 9],
                  [-1, 3, 7, 2, 6, 0, 5, 1, 8, 2],
                  [-4, 1, 2, 3, 6, 5, 0, 7, 9, 9],
                  [-7, 0, 1, 5, 6, 4, 2, 9, 5, 3],
                  [-5, 6, 3, 4, 9, 6, 4, 1, 7, 0],
                  ],
               )


class MinmaxTest(unittest.TestCase):
    """Tests for minmax function."""
    data = list(range(100))
    expected = (0, 99)

    def key(self, n):
        # This must be a monotomically increasing function.
        return n*33 - 11

    def setUp(self):
        self.minmax = stats.minmax
        random.shuffle(self.data)

    def testArgsNoKey(self):
        # Test minmax works with multiple arguments and no key.
        self.assertEqual(self.minmax(*self.data), self.expected)

    def testSequenceNoKey(self):
        # Test minmax works with a single sequence argument and no key.
        self.assertEqual(self.minmax(self.data), self.expected)

    def testIterNoKey(self):
        # Test minmax works with a single iterator argument and no key.
        self.assertEqual(self.minmax(iter(self.data)), self.expected)

    def testArgsKey(self):
        # Test minmax works with multiple arguments and a key function.
        result = self.minmax(*self.data, key=self.key)
        self.assertEqual(result, self.expected)

    def testSequenceKey(self):
        # Test minmax works with a single sequence argument and a key.
        result = self.minmax(self.data, key=self.key)
        self.assertEqual(result, self.expected)

    def testIterKey(self):
        # Test minmax works with a single iterator argument and a key.
        it = iter(self.data)
        self.assertEqual(self.minmax(it, key=self.key), self.expected)

    def testCompareNoKey(self):
        # Test minmax directly against min and max built-ins.
        data = random.sample(range(-5000, 5000), 300)
        expected = (min(data), max(data))
        result = self.minmax(data)
        self.assertEqual(result, expected)
        random.shuffle(data)
        result = self.minmax(iter(data))
        self.assertEqual(result, expected)

    def testCompareKey(self):
        # Test minmax directly against min and max built-ins with a key.
        letters = list('abcdefghij')
        random.shuffle(letters)
        assert len(letters) == 10
        data = [count*letter for (count, letter) in enumerate(letters)]
        random.shuffle(data)
        expected = (min(data, key=len), max(data, key=len))
        result = self.minmax(data, key=len)
        self.assertEqual(result, expected)
        random.shuffle(data)
        result = self.minmax(iter(data), key=len)
        self.assertEqual(result, expected)

    def testFailures(self):
        """Test minmax failure modes."""
        self.assertRaises(TypeError, self.minmax)
        self.assertRaises(ValueError, self.minmax, [])
        self.assertRaises(TypeError, self.minmax, 1)


# -- Tests for private functions --

class CountIterTest(unittest.TestCase):
    # Test the _countiter utility class.

    def test_is_iter(self):
        it = stats._countiter('')
        self.assertTrue(hasattr(it, '__next__'))
        self.assertTrue(hasattr(it, '__iter__'))
        self.assertTrue(it is iter(it))

    def test_iteration(self):
        it = stats._countiter('abcd')
        self.assertEqual(next(it), 'a')
        self.assertEqual(next(it), 'b')
        self.assertEqual(list(it), list('cd'))
        self.assertRaises(StopIteration, next, it)

    def test_count(self):
        it = stats._countiter(range(4))
        self.assertEqual(it.count, 0)
        next(it)
        self.assertEqual(it.count, 1)
        next(it)
        self.assertEqual(it.count, 2)
        list(it)
        self.assertEqual(it.count, 4)


class IsNumericTest(unittest.TestCase):
    # Test the _is_numeric utility function.

    def test_ints(self):
        class MyInt(int): pass
        for n in (-100, -1, 0, 1, 2, 99, 9999):
            self.assertTrue(stats._is_numeric(n))
            self.assertTrue(stats._is_numeric(MyInt(n)))

    def test_bools(self):
        assert isinstance(True, int)
        self.assertTrue(stats._is_numeric(True))
        self.assertTrue(stats._is_numeric(False))

    def test_floats(self):
        class MyFloat(float): pass
        for x in (-123.456, -1.0, 0.0, 1.2e-20, 1.0, 2.5, 4.7e99):
            self.assertTrue(stats._is_numeric(x))
            self.assertTrue(stats._is_numeric(MyFloat(x)))

    def test_complex(self):
        class MyComplex(complex): pass
        for z in (-1+2j, 1+3j, -1-1j):
            self.assertTrue(stats._is_numeric(z))
            self.assertTrue(stats._is_numeric(MyComplex(z)))

    def test_fractions(self):
        self.assertTrue(stats._is_numeric(Fraction(2, 3)))

    def test_decimals(self):
        self.assertTrue(stats._is_numeric(Decimal("2.3456")))

    def test_inf(self):
        self.assertTrue(stats._is_numeric(float('inf')))
        self.assertTrue(stats._is_numeric(float('-inf')))
        self.assertTrue(stats._is_numeric(Decimal('inf')))
        self.assertTrue(stats._is_numeric(Decimal('-inf')))

    def test_nan(self):
        self.assertTrue(stats._is_numeric(float('nan')))
        self.assertTrue(stats._is_numeric(Decimal('nan')))

    def test_non_numbers(self):
        for obj in (None, object(), set([4]), '1', '2.5', [3], (4,5), {2:3}):
            self.assertFalse(stats._is_numeric(obj))


class VSMapTest(unittest.TestCase):
    # Test private function _vsmap

    def test_vmap(self):
        class MyList(list): pass
        data = [1, 2, 3]
        expected = [2, 3, 4]
        self.assertEqual(stats._vsmap(lambda x: x+1, data), expected)
        self.assertEqual(stats._vsmap(lambda x: x+1, MyList(data)), expected)

    def test_smap(self):
        self.assertEqual(stats._vsmap(lambda x: x+1, 4), 5)
        self.assertEqual(stats._vsmap(len, (1,1,1)), 3)
        self.assertEqual(stats._vsmap(len, set([2,3,4])), 3)

    def test_assertion(self):
        vsmap = stats._vsmap
        self.assertRaises(AssertionError, vsmap, len, 'a', lambda x: False)
        self.assertRaises(AssertionError, vsmap, len, ['a'], lambda x: False)
        self.assertEqual(vsmap(len, 'a', lambda x: True), 1)
        self.assertEqual(vsmap(len, ['a'], lambda x: True), [1])



# === Run tests ===

def test_main():
    # support.run_unittest(...list tests...)
    unittest.main()


if __name__ == '__main__':
    test_main()

