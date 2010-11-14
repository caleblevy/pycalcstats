#!/usr/bin/env python3
"""Test suite for stats.py

Runs:
    doctests from the stats module
    tests from the examples text file (if any)
    unit tests in this module
    a limited test for uncollectable objects

"""

import collections
import doctest
import gc
import inspect
import itertools
import math
import os
import pickle
import random
import sys
import unittest
import zipfile


# Reminder to myself that this has to be run under Python3.
if sys.version < "3.0":
    raise RuntimeError("run this under Python3")


# Adjust the path to find the module being tested.
import main_test as mt  # This module.
loc = os.path.split(mt.__file__)[0]
parent = os.path.split(loc)[0]
sys.path.append(os.path.join(parent, 'src'))
del mt, loc, parent


# Modules being tested.
import stats
import _stats_quantiles


# Test internal functions in _stats_quantiles
# -------------------------------------------

class RoundTest(unittest.TestCase):
    UP = _stats_quantiles.UP
    DOWN = _stats_quantiles.DOWN
    EVEN = _stats_quantiles.EVEN

    def testRoundDown(self):
        f = _stats_quantiles.round
        self.assertEquals(f(1.4, self.DOWN), 1)
        self.assertEquals(f(1.5, self.DOWN), 1)
        self.assertEquals(f(1.6, self.DOWN), 2)
        self.assertEquals(f(2.4, self.DOWN), 2)
        self.assertEquals(f(2.5, self.DOWN), 2)
        self.assertEquals(f(2.6, self.DOWN), 3)

    def testRoundUp(self):
        f = _stats_quantiles.round
        self.assertEquals(f(1.4, self.UP), 1)
        self.assertEquals(f(1.5, self.UP), 2)
        self.assertEquals(f(1.6, self.UP), 2)
        self.assertEquals(f(2.4, self.UP), 2)
        self.assertEquals(f(2.5, self.UP), 3)
        self.assertEquals(f(2.6, self.UP), 3)

    def testRoundEven(self):
        f = _stats_quantiles.round
        self.assertEquals(f(1.4, self.EVEN), 1)
        self.assertEquals(f(1.5, self.EVEN), 2)
        self.assertEquals(f(1.6, self.EVEN), 2)
        self.assertEquals(f(2.4, self.EVEN), 2)
        self.assertEquals(f(2.5, self.EVEN), 2)
        self.assertEquals(f(2.6, self.EVEN), 3)


# Miscellaneous tests
# -------------------

class GlobalsTest(unittest.TestCase):
    # Test the state and/or existence of globals.
    def testMeta(self):
        # Test existence of metadata.
        attrs = ("__doc__ __version__ __date__ __author__"
                 " __author_email__ __all__").split()
        for meta in attrs:
            self.failUnless(hasattr(stats, meta), "missing %s" % meta)

    def testCheckAll(self):
        # Check everything in __all__ exists.
        for name in stats.__all__:
            self.failUnless(hasattr(stats, name))


class CompareAgainstExternalResultsTest(unittest.TestCase):
    # Test the results we generate against some numpy equivalents.
    places = 8

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        # Read data from external test data file.
        # (In this case, produced by numpy and Python 2.5.)
        zf = zipfile.ZipFile('test_data.zip', 'r')
        self.data = pickle.loads(zf.read('data.pkl'))
        self.expected = pickle.loads(zf.read('results.pkl'))
        zf.close()

    # FIXME assertAlmostEquals is not really the right way to do these
    # tests, as decimal places != significant figures.
    def testSum(self):
        result = stats.sum(self.data)
        expected = self.expected['sum']
        n = int(math.log(result, 10))  # Yuck.
        self.assertAlmostEqual(result, expected, places=self.places-n)

    def testProduct(self):
        result = stats.product(self.data)
        expected = self.expected['product']
        self.assertAlmostEqual(result, expected, places=self.places)

    def testMean(self):
        result = stats.mean(self.data)
        expected = self.expected['mean']
        self.assertAlmostEqual(result, expected, places=self.places)

    def testRange(self):
        result = stats.range(self.data)
        expected = self.expected['range']
        self.assertAlmostEqual(result, expected, places=self.places)

    def testMidrange(self):
        result = stats.midrange(self.data)
        expected = self.expected['midrange']
        self.assertAlmostEqual(result, expected, places=self.places)

    def testPStdev(self):
        result = stats.pstdev(self.data)
        expected = self.expected['pstdev']
        self.assertAlmostEqual(result, expected, places=self.places)

    def testPVar(self):
        result = stats.pvariance(self.data)
        expected = self.expected['pvariance']
        self.assertAlmostEqual(result, expected, places=self.places)


# Test helper and utility functions
# ---------------------------------

class SortedDataDecoratorTest(unittest.TestCase):
    """Test that the sorted_data decorator works correctly."""
    def testDecorator(self):
        @stats.sorted_data
        def f(data):
            return data

        values = random.sample(range(1000), 100)
        result = f(values)
        self.assertEquals(result, sorted(values))


class MultivariateDecoratorTest(unittest.TestCase):
    """Test that the multivariate decorator works correctly."""
    def get_decorated_result(self, *args):
        @stats.multivariate
        def f(data):
            return list(data)
        return f(*args)

    def test_xy_apart(self):
        xdata = range(8)
        ydata = (2**i for i in xdata)
        expected = [(i, 2**i) for i in range(8)]
        result = self.get_decorated_result(xdata, ydata)
        self.assertEquals(result, expected)

    def test_xy_together(self):
        xdata = range(64)
        ydata = (2*i-3 for i in xdata)
        expected = [(i, 2*i-3) for i in range(64)]
        result = self.get_decorated_result(zip(xdata, ydata))
        self.assertEquals(result, expected)

    def test_x_alone(self):
        xdata = [2, 4, 6, 8]
        expected = [(2, None), (4, None), (6, None), (8, None)]
        result = self.get_decorated_result(xdata)
        self.assertEquals(result, expected)


class MinmaxTest(unittest.TestCase):
    """Tests for minmax function."""
    data = list(range(100))
    expected = (0, 99)

    def key(self, n):
        # Tests assume this is a monotomically increasing function.
        return n*33 - 11

    def setUp(self):
        random.shuffle(self.data)

    def testArgsNoKey(self):
        """Test minmax works with multiple arguments and no key."""
        self.assertEquals(stats.minmax(*self.data), self.expected)

    def testSequenceNoKey(self):
        """Test minmax works with a single sequence argument and no key."""
        self.assertEquals(stats.minmax(self.data), self.expected)

    def testIterNoKey(self):
        """Test minmax works with a single iterator argument and no key."""
        self.assertEquals(stats.minmax(iter(self.data)), self.expected)

    def testArgsKey(self):
        """Test minmax works with multiple arguments and a key function."""
        result = stats.minmax(*self.data, key=self.key)
        self.assertEquals(result, self.expected)

    def testSequenceKey(self):
        """Test minmax works with a single sequence argument and a key."""
        result = stats.minmax(self.data, key=self.key)
        self.assertEquals(result, self.expected)

    def testIterKey(self):
        """Test minmax works with a single iterator argument and a key."""
        it = iter(self.data)
        self.assertEquals(stats.minmax(it, key=self.key), self.expected)

    def testCompareNoKey(self):
        """Test minmax directly against min and max built-ins."""
        data = random.sample(range(-5000, 5000), 300)
        expected = (min(data), max(data))
        result = stats.minmax(data)
        self.assertEquals(result, expected)
        random.shuffle(data)
        result = stats.minmax(iter(data))
        self.assertEquals(result, expected)

    def testCompareKey(self):
        """Test minmax directly against min and max built-ins with a key."""
        letters = list('abcdefghij')
        random.shuffle(letters)
        assert len(letters) == 10
        data = [count*letter for (count, letter) in enumerate(letters)]
        random.shuffle(data)
        expected = (min(data, key=len), max(data, key=len))
        result = stats.minmax(data, key=len)
        self.assertEquals(result, expected)
        random.shuffle(data)
        result = stats.minmax(iter(data), key=len)
        self.assertEquals(result, expected)

    def testFailures(self):
        """Test minmax failure modes."""
        self.assertRaises(TypeError, stats.minmax)
        self.assertRaises(ValueError, stats.minmax, [])
        self.assertRaises(TypeError, stats.minmax, 1)


class AddPartialTest(unittest.TestCase):
    def testInplace(self):
        # Test that add_partial modifies list in place and returns None.
        L = []
        result = stats.add_partial(1.5, L)
        self.assertEquals(L, [1.5])
        self.assert_(result is None)


class AsSequenceTest(unittest.TestCase):
    def testIdentity(self):
        data = [1, 2, 3]
        self.assert_(stats.as_sequence(data) is data)
        data = tuple(data)
        self.assert_(stats.as_sequence(data) is data)

    def testSubclass(self):
        # Helper function.
        def make_subclass(kind):
            class Subclass(kind):
                pass
            return Subclass

        for cls in (tuple, list):
            subcls = make_subclass(cls)
            data = subcls([1, 2, 3])
            assert type(data) is not cls
            assert issubclass(type(data), cls)
            self.assert_(stats.as_sequence(data) is data)

    def testOther(self):
        data = range(20)
        assert type(data) is not list
        result = stats.as_sequence(data)
        self.assertEquals(result, list(data))
        self.assert_(isinstance(result, list))


class CombineXYDataTest(unittest.TestCase):
    def test_xy_together(self):
        xydata = [(1, 2), (3, 4), (5, 6)]
        expected = xydata[:]
        result = stats.combine_xydata(xydata)
        self.assertEquals(list(result), expected)

    def test_xy_apart(self):
        xdata = [1, 3, 5]
        ydata = [2, 4, 6]
        expected = list(zip(xdata, ydata))
        result = stats.combine_xydata(xdata, ydata)
        self.assertEquals(list(result), expected)

    def test_x_alone(self):
        xdata = [1, 3, 5]
        expected = list(zip(xdata, [None]*len(xdata)))
        result = stats.combine_xydata(xdata)
        self.assertEquals(list(result), expected)


class ValidateIntTest(unittest.TestCase):
    def testIntegers(self):
        for n in (-100, -1, 0, 1, 23, 42, 2**80):
            stats._validate_int(n)

    def testGoodFloats(self):
        for n in (-100.0, -1.0, 0.0, 1.0, 23.0, 42.0, 1.23456e18):
            stats._validate_int(n)

    def testBadFloats(self):
        for x in (-100.1, -1.2, 0.3, 1.4, 23.5, 42.6, float('nan')):
            self.assertRaises(ValueError, stats._validate_int, x)

    def testBadInfinity(self):
        for x in (float('-inf'), float('inf')):
            self.assertRaises(OverflowError, stats._validate_int, x)

    def testBadTypes(self):
        for obj in ("a", "1", None, []):
            self.assertRaises((ValueError, TypeError),
                stats._validate_int, obj)


# Tests for univariate statistics: means and averages
# ---------------------------------------------------

class MeanTest(unittest.TestCase):
    data = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
    expected = 5.5
    func = stats.mean

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        # Black magic to force self.func to be a function rather
        # than a method.
        self.func = self.__class__.func

    def setUp(self):
        random.shuffle(self.data)

    def myAssertEquals(self, a, b, **kwargs):
        if hasattr(self, 'delta'):
            diff = abs(a-b)
            self.assertLessEqual(diff, self.delta, **kwargs)
        else:
            self.assertEquals(a, b, **kwargs)

    def testEmpty(self):
        self.assertRaises(ValueError, self.func, [])

    def testSeq(self):
        self.myAssertEquals(self.func(self.data), self.expected)

    def testBigData(self):
        data = [x + 1e9 for x in self.data]
        expected = self.expected + 1e9
        self.myAssertEquals(self.func(data), expected)

    def testIter(self):
        self.myAssertEquals(self.func(iter(self.data)), self.expected)

    def testSingleton(self):
        for x in self.data:
            self.myAssertEquals(self.func([x]), x)


class HarmonicMeanTest(MeanTest):
    func = stats.harmonic_mean
    expected = 3.4995090404755
    delta = 1e-8

    def testBigData(self):
        data = [x + 1e9 for x in self.data]
        expected = 1000000005.5  # Calculated with HP-48GX
        diff = abs(self.func(data) - expected)
        self.assertLessEqual(diff, 1e-6)


class GeometricMeanTest(MeanTest):
    func = stats.geometric_mean
    expected = 4.56188290183
    delta = 1e-11

    def testBigData(self):
        data = [x + 1e9 for x in self.data]
        # HP-48GX calculates this as 1000000005.48
        expected = 1000000005.5
        self.assertEquals(self.func(data), expected)

    def testNegative(self):
        data = [1.0, 2.0, -3.0, 4.0]
        assert any(x < 0.0 for x in data)
        self.assertRaises(ValueError, self.func, data)

    def testZero(self):
        data = [1.0, 2.0, 0.0, 4.0]
        assert any(x == 0.0 for x in data)
        self.assertEquals(self.func(data), 0.0)


class QuadraticMeanTest(MeanTest):
    func = stats.quadratic_mean
    expected = 6.19004577259
    delta = 1e-11

    def testBigData(self):
        data = [x + 1e9 for x in self.data]
        expected = 1000000005.5  # Calculated with HP-48GX
        self.assertEquals(self.func(data), expected)

    def testNegative(self):
        data = [-x for x in self.data]
        self.myAssertEquals(self.func(data), self.expected)


class MedianTest(MeanTest):
    func = stats.median

    def testSeq(self):
        assert len(self.data) % 2 == 1
        MeanTest.testSeq(self)

    def testEven(self):
        data = self.data[:] + [0.0]
        self.assertEquals(self.func(data), 4.95)

    def testSorting(self):
        """Test that median doesn't sort in place."""
        data = [2, 4, 1, 3]
        assert data != sorted(data)
        save = data[:]
        assert save is not data
        _ = stats.median(data)
        self.assertEquals(data, save)


class ModeTest(MeanTest):
    data = [1.1, 2.2, 2.2, 3.3, 4.4, 5.5, 5.5, 5.5, 5.5, 6.6, 6.6, 7.7, 8.8]
    func = stats.mode
    expected = 5.5

    def testModeless(self):
        data = list(set(self.data))
        random.shuffle(data)
        self.assertRaises(ValueError, self.func, data)

    def testBimodal(self):
        data = self.data[:]
        n = data.count(self.expected)
        data.extend([6.6]*(n-data.count(6.6)))
        assert data.count(6.6) == n
        self.assertRaises(ValueError, self.func, data)


class MidrangeTest(MeanTest):
    func = stats.midrange

    def testMidrange(self):
        self.assertEquals(stats.midrange([1.0, 2.5]), 1.75)
        self.assertEquals(stats.midrange([1.0, 2.0, 4.0]), 2.5)
        self.assertEquals(stats.midrange([2.0, 4.0, 1.0]), 2.5)


class TrimeanTest(unittest.TestCase):
    data = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
    expected = 5.5

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.func = stats.trimean

    def testFailures(self):
        self.assertRaises(ValueError, self.func, [])
        self.assertRaises(ValueError, self.func, [1])
        self.assertRaises(ValueError, self.func, [1, 2])

    def generic_sequence_test(self, data, n, expected):
        assert len(data)%4 == n
        random.shuffle(data)
        result = self.func(data)
        self.assertEquals(result, expected)
        data = [x + 1e9 for x in data]
        result = self.func(data)
        self.assertEquals(result, expected+1e9)

    def testSeq0(self):
        data = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8]
        expected = ((2.2+3.3)/2 + 4.4 + 5.5 + (6.6+7.7)/2)/4
        self.generic_sequence_test(data, 0, expected)

    def testSeq1(self):
        data = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
        expected = (3.3 + 5.5*2 + 7.7)/4
        self.generic_sequence_test(data, 1, expected)

    def testSeq2(self):
        data = [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
        expected = (2.2 + 4.4 + 5.5 + 7.7)/4
        self.generic_sequence_test(data, 2, expected)

    def testSeq3(self):
        data = [-1.1, 0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
        expected = ((1.1+2.2)/2 + 4.4*2 + (6.6+7.7)/2)/4
        self.generic_sequence_test(data, 3, expected)

    def testIter(self):
        data = [1.1, 3.3, 4.4, 6.6, 7.7, 9.9]
        expected = (3.3 + 4.4 + 6.6 + 7.7)/4
        self.assertEquals(self.func(iter(data)), expected)


# Tests for moving averages
# -------------------------

class RunningAverageTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.func = stats.running_average

    def testGenerator(self):
        # Test that function is a generator.
        self.assert_(inspect.isgeneratorfunction(self.func))

    def testFinal(self):
        # Test the final result has the expected value.
        data = [3.2*i - 12.3 for i in range(0, 35, 3)]
        random.shuffle(data)
        expected = stats.mean(data)
        results = list(self.func(data))
        self.assertAlmostEquals(results[-1], expected)


class WeightedRunningAverageTest(RunningAverageTest):
    def __init__(self, *args, **kwargs):
        RunningAverageTest.__init__(self, *args, **kwargs)
        self.func = stats.weighted_running_average

    def testFinal(self):
        # Test the final result has the expected value.
        data = [64, 32, 16, 8, 4, 2, 1]
        results = list(self.func(data))
        self.assertEquals(results[-1], 4)

class SimpleMovingAverageTest(RunningAverageTest):
    def __init__(self, *args, **kwargs):
        RunningAverageTest.__init__(self, *args, **kwargs)
        self.func = stats.simple_moving_average

    def testFinal(self):
        # Test the final result has the expected value.
        data = [1, 2, 3, 4, 5, 6, 7, 8]
        results = list(self.func(data))
        self.assertEquals(results[-1], 7.0)


# Test quantiles
# --------------

class DrMathTests(unittest.TestCase):
    # Sample data for testing quartiles taken from Dr Math page:
    # http://mathforum.org/library/drmath/view/60969.html
    # Q2 values are not cheked in this test.
    A = range(1, 9)
    B = range(1, 10)
    C = range(1, 11)
    D = range(1, 12)

    def testInclusive(self):
        f = _stats_quantiles.inclusive
        q1, _, q3 = f(self.A)
        self.assertEquals(q1, 2.5)
        self.assertEquals(q3, 6.5)
        q1, _, q3 = f(self.B)
        self.assertEquals(q1, 3.0)
        self.assertEquals(q3, 7.0)
        q1, _, q3 = f(self.C)
        self.assertEquals(q1, 3.0)
        self.assertEquals(q3, 8.0)
        q1, _, q3 = f(self.D)
        self.assertEquals(q1, 3.5)
        self.assertEquals(q3, 8.5)

    def testExclusive(self):
        f = _stats_quantiles.exclusive
        q1, _, q3 = f(self.A)
        self.assertEquals(q1, 2.5)
        self.assertEquals(q3, 6.5)
        q1, _, q3 = f(self.B)
        self.assertEquals(q1, 2.5)
        self.assertEquals(q3, 7.5)
        q1, _, q3 = f(self.C)
        self.assertEquals(q1, 3.0)
        self.assertEquals(q3, 8.0)
        q1, _, q3 = f(self.D)
        self.assertEquals(q1, 3.0)
        self.assertEquals(q3, 9.0)

    def testMS(self):
        f = _stats_quantiles.ms
        q1, _, q3 = f(self.A)
        self.assertEquals(q1, 2)
        self.assertEquals(q3, 7)
        q1, _, q3 = f(self.B)
        self.assertEquals(q1, 3)
        self.assertEquals(q3, 7)
        q1, _, q3 = f(self.C)
        self.assertEquals(q1, 3)
        self.assertEquals(q3, 8)
        q1, _, q3 = f(self.D)
        self.assertEquals(q1, 3)
        self.assertEquals(q3, 9)

    def testMinitab(self):
        f = _stats_quantiles.minitab
        q1, _, q3 = f(self.A)
        self.assertEquals(q1, 2.25)
        self.assertEquals(q3, 6.75)
        q1, _, q3 = f(self.B)
        self.assertEquals(q1, 2.5)
        self.assertEquals(q3, 7.5)
        q1, _, q3 = f(self.C)
        self.assertEquals(q1, 2.75)
        self.assertEquals(q3, 8.25)
        q1, _, q3 = f(self.D)
        self.assertEquals(q1, 3.0)
        self.assertEquals(q3, 9.0)

    def testExcel(self):
        f = _stats_quantiles.excel
        q1, _, q3 = f(self.A)
        self.assertEquals(q1, 2.75)
        self.assertEquals(q3, 6.25)
        q1, _, q3 = f(self.B)
        self.assertEquals(q1, 3.0)
        self.assertEquals(q3, 7.0)
        q1, _, q3 = f(self.C)
        self.assertEquals(q1, 3.25)
        self.assertEquals(q3, 7.75)
        q1, _, q3 = f(self.D)
        self.assertEquals(q1, 3.5)
        self.assertEquals(q3, 8.5)


class QuartileAliases(unittest.TestCase):
    allowed_methods = set(_stats_quantiles.QUARTILE_MAP.keys())

    def testAliasesMapping(self):
        # Test that the quartile function exposes a mapping of aliases.
        self.assert_(hasattr(stats.quartiles, 'aliases'))
        aliases = stats.quartiles.aliases
        self.assert_(isinstance(aliases, collections.Mapping))
        self.assert_(aliases)

    def testAliasesValues(self):
        for method in stats.quartiles.aliases.values():
            self.assert_(method in self.allowed_methods)


class QuartileTest(unittest.TestCase):
    func = stats.quartiles
    # Methods to be tested.
    methods = [0, 1]

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        # Black magic to force self.func to be a function rather
        # than a method.
        self.func = self.__class__.func

    # Helper methods:

    def compare_sorted_with_unsorted(self, n, method):
        data = list(range(n))
        result1 = self.func(data, method)
        random.shuffle(data)
        result2 = self.func(data, method)
        self.assertEquals(result1, result2)

    def compare_types(self, n, method):
        data = range(n)
        result1 = self.func(data, method)
        data = list(data)
        result2 = self.func(data, method)
        result3 = self.func(tuple(data), method)
        result4 = self.func(iter(data), method)
        self.assertEquals(result1, result2)
        self.assertEquals(result1, result3)
        self.assertEquals(result1, result4)

    def expect_failure(self, data):
        self.assertRaises(ValueError, self.func, data)
        for method in self.methods:
            self.assertRaises(ValueError, self.func, data, method)

    # Generic tests that don't care about the specific values:

    def testNoSorting(self):
        """Test that quartiles doesn't sort in place."""
        data = [2, 4, 1, 3, 0, 5]
        save = data[:]
        assert save is not data
        assert data != sorted(data)
        for method in self.methods:
            _ = self.func(data, method)
            self.assertEquals(data, save)

    def testTooFewItems(self):
        for data in ([], [1], [1, 2]):
            self.expect_failure(data)

    def testSorted(self):
        # Test that sorted and unsorted data give the same results.
        for n in (8, 9, 10, 11):  # n%4 -> 0...4
            for method in self.methods:
                self.compare_sorted_with_unsorted(n, method)

    def testIter(self):
        # Test that iterators and sequences give the same result.
        for n in (8, 9, 10, 11):  # n%4 -> 0...4
            for method in self.methods:
                self.compare_types(n, method)

    # Tests where we check for the correct result.

    def testInclusive(self):
        # Test the inclusive method of calculating quartiles.
        f = self.func
        m = 0
        self.assertEquals(f([0, 1, 2], m), (0.5, 1, 1.5))
        #--
        self.assertEquals(f([0, 1, 2, 3], m), (0.5, 1.5, 2.5))
        self.assertEquals(f([0, 1, 2, 3, 4], m), (1, 2, 3))
        self.assertEquals(f([0, 1, 2, 3, 4, 5], m), (1, 2.5, 4))
        self.assertEquals(f([0, 1, 2, 3, 4, 5, 6], m), (1.5, 3, 4.5))
        #--
        self.assertEquals(f(range(1, 9), m), (2.5, 4.5, 6.5))
        self.assertEquals(f(range(1, 10), m), (3, 5, 7))
        self.assertEquals(f(range(1, 11), m), (3, 5.5, 8))
        self.assertEquals(f(range(1, 12), m), (3.5, 6, 8.5))
        #--
        self.assertEquals(f(range(1, 13), m), (3.5, 6.5, 9.5))
        self.assertEquals(f(range(1, 14), m), (4, 7, 10))
        self.assertEquals(f(range(1, 15), m), (4, 7.5, 11))
        self.assertEquals(f(range(1, 16), m), (4.5, 8, 11.5))

    def testExclusive(self):
        # Test the exclusive method of calculating quartiles.
        f = self.func
        m = 1
        #self.assertEquals(f([0, 1, 2], m), (0, 1, 2))
        #--
        self.assertEquals(f([0, 1, 2, 3], m), (0.5, 1.5, 2.5))
        self.assertEquals(f([0, 1, 2, 3, 4], m), (0.5, 2, 3.5))
        self.assertEquals(f([0, 1, 2, 3, 4, 5], m), (1, 2.5, 4))
        self.assertEquals(f([0, 1, 2, 3, 4, 5, 6], m), (1, 3, 5))
        #--
        self.assertEquals(f(range(1, 9), m), (2.5, 4.5, 6.5))
        self.assertEquals(f(range(1, 10), m), (2.5, 5, 7.5))
        self.assertEquals(f(range(1, 11), m), (3, 5.5, 8))
        self.assertEquals(f(range(1, 12), m), (3, 6, 9))
        #--
        self.assertEquals(f(range(1, 13), m), (3.5, 6.5, 9.5))
        self.assertEquals(f(range(1, 14), m), (3.5, 7, 10.5))
        self.assertEquals(f(range(1, 15), m), (4, 7.5, 11))
        self.assertEquals(f(range(1, 16), m), (4, 8, 12))

    def notestBig(self):
        data = list(range(1000, 2000))
        assert len(data) == 1000
        assert len(data)%4 == 0
        random.shuffle(data)
        self.assertEquals(self.func(data), (1249.5, 1499.5, 1749.5))
        data.append(2000)
        random.shuffle(data)
        self.assertEquals(self.func(data), (1249.5, 1500, 1750.5))
        data.append(2001)
        random.shuffle(data)
        self.assertEquals(self.func(data), (1250, 1500.5, 1751))
        data.append(2002)
        random.shuffle(data)
        self.assertEquals(self.func(data), (1250, 1501, 1752))


class QuantileTest(unittest.TestCase):
    func = stats.quantile

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        # Black magic to force self.func to be a function rather
        # than a method.
        self.func = self.__class__.func

    def testSorting(self):
        # Test that quantile doesn't sort in place.
        data = [2, 4, 1, 3, 0, 5]
        assert data != sorted(data)
        save = data[:]
        assert save is not data
        _ = self.func(data, 0.9)
        self.assertEquals(data, save)

    def testQuantileArgOutOfRange(self):
        data = [1, 2, 3, 4]
        self.assertRaises(ValueError, self.func, data, -0.1)
        self.assertRaises(ValueError, self.func, data, 1.1)

    def testTooFewItems(self):
        self.assertRaises(ValueError, self.func, [], 0.1)
        self.assertRaises(ValueError, self.func, [1], 0.1)

    def testUnsorted(self):
        data = [3, 4, 2, 1, 0, 5]
        assert data != sorted(data)
        self.assertEquals(self.func(data, 0.1), 0.5)
        self.assertEquals(self.func(data, 0.9), 4.5)

    def testIter(self):
        self.assertEquals(self.func(range(12), 0.3), 3.3)

    def testUnitInterval(self):
        data = [0, 1]
        for f in (0.01, 0.1, 0.2, 0.25, 0.5, 0.55, 0.8, 0.9, 0.99):
            self.assertEquals(self.func(data, f), f)


class DecileTest(unittest.TestCase):
    pass


class PercentileTest(unittest.TestCase):
    pass


# Test spread statistics
# ----------------------

class PVarianceTest(unittest.TestCase):
    func = stats.pvariance
    data = (4.0, 7.0, 13.0, 16.0)
    expected = 22.5  # Exact population variance.

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        # Force self.func to be a function rather than a method.
        self.func = self.__class__.func

    def testEmptyFailure(self):
        for data in ([], (), iter([])):
            self.assertRaises(ValueError, self.func, data)

    def test_small(self):
        self.assertEquals(self.func(self.data), self.expected)

    def test_big(self):
        data = [x + 1e6 for x in self.data]
        self.assertEquals(self.func(data), self.expected)

    def test_huge(self):
        data = [x + 1e9 for x in self.data]
        self.assertEquals(self.func(data), self.expected)


class VarianceTest(PVarianceTest):
    func = stats.variance
    expected = 30.0  # Exact sample variance.

    def testSingletonFailure(self):
        for data in ([1], iter([1])):
            self.assertRaises(ValueError, self.func, data)


class PStdevTest(PVarianceTest):
    func = stats.pstdev
    expected = math.sqrt(22.5)  # Exact population stdev.


class StdevTest(VarianceTest):
    func = stats.stdev
    expected = math.sqrt(30.0)  # Exact sample stdev.


class PVariance1Test(PVarianceTest):
    func = stats.pvariance1


class Variance1Test(VarianceTest):
    func = stats.variance1


class PStdev1Test(PStdevTest):
    func = stats.pstdev1


class Stdev1Test(StdevTest):
    func = stats.stdev1


class RangeTest(unittest.TestCase):
    def testFailure(self):
        self.assertRaises(ValueError, stats.range, [])
        self.assertRaises(ValueError, stats.range, iter([]))

    def testSingleton(self):
        for x in (-3.1, 0.0, 4.2, 1.789e12):
            self.assertEquals(stats.range([x]), 0)

    def generate_data_sets(self):
        """Yield 2-tuples of (data, expected range)."""
        # data should be a list, tuple or builtin range object.
        yield ([1, 5], 4)
        yield ((7, 2), 5)
        yield ((3.25, 7.5), 4.25)
        yield ([1, 4, 8], 7)
        data = list(range(500))
        random.shuffle(data)
        for shift in (0, 0.5, 1234.567, -1000, 1e6, 1e9):
            d = [x + shift for x in data]
            yield (d, 499)

    def testSequence(self):
        for data, expected in self.generate_data_sets():
            self.assertEquals(stats.range(data), expected)

    def testIterator(self):
        for data, expected in self.generate_data_sets():
            self.assertEquals(stats.range(iter(data)), expected)


class IQRTest(unittest.TestCase):
    pass


class AverageDeviationTest(unittest.TestCase):
    pass


class MedianAverageDeviationTest(unittest.TestCase):
    pass


# Test other moments
# ------------------

class PearsonModeSkewnessTest(unittest.TestCase):
    pass


class SkewTest(unittest.TestCase):
    pass


class KurtosisTest(unittest.TestCase):
    pass


# Test multivariate statistics
# ----------------------------

class QCorrTest(unittest.TestCase):
    pass


class CorrTest(unittest.TestCase):
    pass


class Corr1Test(CorrTest):
    pass


class PCovTest(unittest.TestCase):
    pass


class CovTest(PCovTest):
    pass


class ErrSumSqTest(unittest.TestCase):
    pass


class LinrTest(unittest.TestCase):
    pass


# Test sums and products
# ----------------------

class SumTest(unittest.TestCase):
    pass


class ProductTest(unittest.TestCase):
    pass


class SumSqTest(SumTest):
    pass


class SxxTest(unittest.TestCase):
    pass


class SyyTest(SxxTest):
    pass


class SxyTest(SxxTest):
    pass


class XSumsTest(unittest.TestCase):
    pass


class XYSumsTest(unittest.TestCase):
    pass


# Test partitioning and binning
# -----------------------------



# Test trimming
# -------------



# Test other statistical formulae
# -------------------------------

class StErrMeanTest(unittest.TestCase):
    pass


# Test statistics of circular quantities
# --------------------------------------

class CircularMeanTest(unittest.TestCase):

    def testDefaultDegrees(self):
        # Test that degrees are the default.
        data = [355, 5, 15, 320, 45]
        theta = stats.circular_mean(data)
        phi = stats.circular_mean(data, True)
        assert stats.circular_mean(data, False) != theta
        self.assertEquals(theta, phi)

    def testRadians(self):
        # Test that degrees and radians (usually) give different results.
        data = [355, 5, 15, 320, 45]
        a = stats.circular_mean(data, True)
        b = stats.circular_mean(data, False)
        self.assertNotEquals(a, b)

    def testEmpty(self):
        self.assertRaises(ValueError, stats.circular_mean, [])

    def testSingleton(self):
        for x in (-1.0, 0.0, 1.0, 3.0):
            self.assertEquals(stats.circular_mean([x], False), x)
            self.assertAlmostEquals(
                stats.circular_mean([x], True), x, places=12
                )

    def testNegatives(self):
        data1 = [355, 5, 15, 320, 45]
        theta = stats.circular_mean(data1)
        data2 = [d-360 if d > 180 else d for d in data1]
        phi = stats.circular_mean(data2)
        self.assertAlmostEquals(theta, phi, places=12)

    def testIter(self):
        theta = stats.circular_mean(iter([355, 5, 15]))
        self.assertAlmostEquals(theta, 5.0, places=12)

    def testSmall(self):
        places = 12
        t = stats.circular_mean([0, 360])
        self.assertEquals(round(t, places), 0.0)
        t = stats.circular_mean([10, 20, 30])
        self.assertEquals(round(t, places), 20.0)
        t = stats.circular_mean([355, 5, 15])
        self.assertEquals(round(t, places), 5.0)

    def testFullCircle(self):
        # Test with angle > full circle.
        places = 12
        theta = stats.circular_mean([3, 363])
        self.assertAlmostEquals(theta, 3, places=places)

    def testBig(self):
        places = 12
        pi = math.pi
        # Generate angles between pi/2 and 3*pi/2, with expected mean of pi.
        delta = pi/1000
        data = [pi/2 + i*delta for i in range(1000)]
        data.append(3*pi/2)
        assert data[0] == pi/2
        assert len(data) == 1001
        random.shuffle(data)
        theta = stats.circular_mean(data, False)
        self.assertAlmostEquals(theta, pi, places=places)
        # Now try the same with angles in the first and fourth quadrants.
        data = [0.0]
        for i in range(1, 501):
            data.append(i*delta)
            data.append(2*pi - i*delta)
        assert len(data) == 1001
        random.shuffle(data)
        theta = stats.circular_mean(data, False)
        self.assertAlmostEquals(theta, 0.0, places=places)



# ============================================================================

if __name__ == '__main__':
    # Define a function that prints, or doesn't, according to whether or not
    # we're in (slightly) quiet mode. Note that we always print "skip" and
    # failure messages.
    # FIX ME can we make unittest run silently if there are no errors?
    if '-q' in sys.argv[1:]:
        def pr(s):
            pass
    else:
        def pr(s):
            print(s)
    #
    # Now run the tests.
    #
    gc.collect()
    assert not gc.garbage
    #
    # Run doctests in the stats package.
    #
    failures, tests = doctest.testmod(stats)
    if failures:
        print("Skipping further tests while doctests failing.")
        sys.exit(1)
    else:
        pr("Doctests: failed %d, attempted %d" % (failures, tests))
    #
    # Run doctests in the example text file.
    #
    if os.path.exists('examples.txt'):
        failures, tests = doctest.testfile('examples.txt')
        if failures:
            print("Skipping further tests while doctests failing.")
            sys.exit(1)
        else:
            pr("Example doc tests: failed %d, attempted %d" % (failures, tests))
    else:
        pr('WARNING: No example text file found.')
    #
    # Run unit tests.
    #
    pr("Running unit tests:")
    try:
        unittest.main()
    except SystemExit:
        pass
    #
    # Check for reference leaks.
    #
    gc.collect()
    if gc.garbage:
        print("List of uncollectable garbage:")
        print(gc.garbage)
    else:
        pr("No garbage found.")


