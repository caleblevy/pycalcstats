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


# Module being tested.
import stats


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

    # FIXME test to make sure that things that should be in __all__ are?


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


class MultivariateSplitDecoratorTest(unittest.TestCase):
    """Test that the multivariate split decorator works correctly."""
    def get_split_result(self, *args):
        @stats._Multivariate.split_xydata
        def f(xdata, ydata):
            return (xdata, ydata)
        return f(*args)

    def test_empty(self):
        empty = iter([])
        result = self.get_split_result(empty)
        self.assertEquals(result, ([], []))
        result = self.get_split_result(empty, empty)
        self.assertEquals(result, ([], []))

    def test_xy_apart(self):
        xdata = range(8)
        ydata = [2**i for i in xdata]
        result = self.get_split_result(xdata, ydata)
        self.assertEquals(result, (list(xdata), ydata))

    def test_xy_together(self):
        xydata = [(i, 2**i) for i in range(8)]
        xdata = [x for x,y in xydata]
        ydata = [y for x,y in xydata]
        result = self.get_split_result(xydata)
        self.assertEquals(result, (xdata, ydata))

    def test_x_alone(self):
        xdata = [2, 4, 6, 8]
        result = self.get_split_result(xdata)
        self.assertEquals(result, (xdata, [None]*4))


class MultivariateMergeDecoratorTest(unittest.TestCase):
    """Test that the multivariate merge decorator works correctly."""
    def get_merge_result(self, *args):
        @stats._Multivariate.merge_xydata
        def f(xydata):
            return list(xydata)
        return f(*args)

    def test_empty(self):
        empty = iter([])
        result = self.get_merge_result(empty)
        self.assertEquals(result, [])
        result = self.get_merge_result(empty, empty)
        self.assertEquals(result, [])

    def test_xy_apart(self):
        expected = [(i, 2**i) for i in range(8)]
        xdata = [x for (x,y) in expected]
        ydata = [y for (x,y) in expected]
        result = self.get_merge_result(xdata, ydata)
        self.assertEquals(result, expected)

    def test_xy_together(self):
        expected = [(i, 2**i) for i in range(8)]
        xdata = [x for x,y in expected]
        ydata = [y for x,y in expected]
        result = self.get_merge_result(zip(xdata, ydata))
        self.assertEquals(result, expected)

    def test_x_alone(self):
        xdata = [2, 4, 6, 8]
        expected = [(x, None) for x in xdata]
        result = self.get_merge_result(xdata)
        self.assertEquals(result, expected)


class MergeTest(unittest.TestCase):
    # Test _Multivariate merge function independantly of the decorator.
    def test_empty(self):
        result = stats._Multivariate.merge([])
        self.assertEquals(list(result), [])
        result = stats._Multivariate.merge([], [])
        self.assertEquals(list(result), [])

    def test_xy_together(self):
        xydata = [(1, 2), (3, 4), (5, 6)]
        expected = xydata[:]
        result = stats._Multivariate.merge(xydata)
        self.assertEquals(list(result), expected)

    def test_xy_apart(self):
        xdata = [1, 3, 5]
        ydata = [2, 4, 6]
        expected = list(zip(xdata, ydata))
        result = stats._Multivariate.merge(xdata, ydata)
        self.assertEquals(list(result), expected)

    def test_x_alone(self):
        xdata = [1, 3, 5]
        expected = list(zip(xdata, [None]*len(xdata)))
        result = stats._Multivariate.merge(xdata)
        self.assertEquals(list(result), expected)


class SplitTest(unittest.TestCase):
    # Test _Multivariate split function independantly of the decorator.
    def test_empty(self):
        result = stats._Multivariate.split([])
        self.assertEquals(result, ([], []))
        result = stats._Multivariate.split([], [])
        self.assertEquals(result, ([], []))

    def test_xy_together(self):
        xydata = [(1, 2), (3, 4), (5, 6)]
        expected = ([1, 3, 5], [2, 4, 6])
        result = stats._Multivariate.split(xydata)
        self.assertEquals(result, expected)

    def test_xy_apart(self):
        xdata = [1, 3, 5]
        ydata = [2, 4, 6]
        result = stats._Multivariate.split(xdata, ydata)
        self.assertEquals(result, (xdata, ydata))

    def test_x_alone(self):
        xdata = [1, 3, 5]
        result = stats._Multivariate.split(xdata)
        self.assertEquals(result, (xdata, [None]*3))


class ValidateIntTest(unittest.TestCase):
    def testIntegers(self):
        for n in (-2**100, -100, -1, 0, 1, 23, 42, 2**80, 2**100):
            stats._validate_int(n)

    def testSubclasses(self):
        class MyInt(int):
            pass
        for n in (True, False, MyInt()):
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
        for obj in ("a", "1", [], {}, object(), None):
            self.assertRaises((ValueError, TypeError),
                stats._validate_int, obj)


class RoundTest(unittest.TestCase):
    UP = stats._UP
    DOWN = stats._DOWN
    EVEN = stats._EVEN

    def testRoundDown(self):
        f = stats._round
        self.assertEquals(f(1.4, self.DOWN), 1)
        self.assertEquals(f(1.5, self.DOWN), 1)
        self.assertEquals(f(1.6, self.DOWN), 2)
        self.assertEquals(f(2.4, self.DOWN), 2)
        self.assertEquals(f(2.5, self.DOWN), 2)
        self.assertEquals(f(2.6, self.DOWN), 3)

    def testRoundUp(self):
        f = stats._round
        self.assertEquals(f(1.4, self.UP), 1)
        self.assertEquals(f(1.5, self.UP), 2)
        self.assertEquals(f(1.6, self.UP), 2)
        self.assertEquals(f(2.4, self.UP), 2)
        self.assertEquals(f(2.5, self.UP), 3)
        self.assertEquals(f(2.6, self.UP), 3)

    def testRoundEven(self):
        f = stats._round
        self.assertEquals(f(1.4, self.EVEN), 1)
        self.assertEquals(f(1.5, self.EVEN), 2)
        self.assertEquals(f(1.6, self.EVEN), 2)
        self.assertEquals(f(2.4, self.EVEN), 2)
        self.assertEquals(f(2.5, self.EVEN), 2)
        self.assertEquals(f(2.6, self.EVEN), 3)


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
        assert expected != 1e9  # Avoid catastrophic loss of precision.
        self.myAssertEquals(self.func(data), expected)

    def testIter(self):
        self.myAssertEquals(self.func(iter(self.data)), self.expected)

    def testSingleton(self):
        for x in self.data:
            self.myAssertEquals(self.func([x]), x)

    def testDoubling(self):
        # Average of [a,b,c...z] should be same as for [a,a,b,b,c,c...z,z].
        data = [random.random() for _ in range(1000)]
        a = self.func(data)
        b = self.func(data*2)
        self.assertEquals(a, b)


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


class MedianExtrasTest(MedianTest):

    def testMedianLow(self):
        data = [11, 12, 13, 14, 15, 16, 17, 18]
        assert len(data)%2 == 0
        random.shuffle(data)
        self.assertEquals(self.func(data, -1), 14)

    def testMedianNormal(self):
        data = [11, 12, 13, 14, 15, 16, 17, 18]
        assert len(data)%2 == 0
        random.shuffle(data)
        self.assertEquals(self.func(data, 0), 14.5)

    def testMedianHigh(self):
        data = [11, 12, 13, 14, 15, 16, 17, 18]
        assert len(data)%2 == 0
        random.shuffle(data)
        self.assertEquals(self.func(data, 1), 15)


class ModeTest(MeanTest):
    data = [1.1, 2.2, 2.2, 3.3, 4.4, 5.5, 5.5, 5.5, 5.5, 6.6, 6.6, 7.7, 8.8]
    func = stats.mode
    expected = 5.5

    def testModeless(self):
        data = list(set(self.data))
        random.shuffle(data)
        self.assertRaises(ValueError, self.func, data)

    def testDoubling(self):
        data = [random.random() for _ in range(1000)]
        self.assertRaises(ValueError, self.func, data*2)

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


class MidhingeTest(MedianTest):
    func = stats.midhinge

    def testSingleton(self):
        self.assertRaises(ValueError, self.func, [1])
        self.assertRaises(ValueError, self.func, [1, 2])

    def testMidhinge(self):
        # True hinges occur for n = 4N+5 items, which is 1 modulo 4.
        # We test midhinge on four test data sets, for 1, 2, 3, 0 modulo 4.
        a = [0.1, 0.4, 1.1, 1.4, 2.1, 2.4, 3.1, 3.4, 4.1, 4.4, 5.1, 5.4, 6.1]
        assert len(a) == 4*2 + 5
        b = a + [6.4]
        c = b + [7.1]
        d = c + [7.4]
        for L in (a, b, c, d):
            random.shuffle(L)
        self.assertEquals(round(stats.midhinge(a), 10), 2.9)
        self.assertEquals(stats.midhinge(b), 3.25)
        self.assertEquals(stats.midhinge(c), 3.5)
        self.assertEquals(stats.midhinge(d), 3.75)


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


# Test order statistics
# ---------------------

class DrMathTests(unittest.TestCase):
    # Sample data for testing quartiles taken from Dr Math page:
    # http://mathforum.org/library/drmath/view/60969.html
    # Q2 values are not checked in this test.
    A = range(1, 9)
    B = range(1, 10)
    C = range(1, 11)
    D = range(1, 12)

    def testInclusive(self):
        f = stats._Quartiles.inclusive
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
        f = stats._Quartiles.exclusive
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
        f = stats._Quartiles.ms
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
        f = stats._Quartiles.minitab
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
        f = stats._Quartiles.excel
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


class QuartileAliasesTest(unittest.TestCase):
    allowed_methods = set(stats._Quartiles.QUARTILE_MAP.keys())

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
    # Schemes to be tested.
    schemes = [1, 2, 3, 4, 5, 6]

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        # Force self.func to be a function rather than a method.
        self.func = self.__class__.func

    # Helper methods:

    def compare_sorted_with_unsorted(self, n, scheme):
        data = list(range(n))
        result1 = self.func(data, scheme)
        random.shuffle(data)
        result2 = self.func(data, scheme)
        self.assertEquals(result1, result2)

    def compare_types(self, n, scheme):
        data = range(n)
        result1 = self.func(data, scheme)
        data = list(data)
        result2 = self.func(data, scheme)
        result3 = self.func(tuple(data), scheme)
        result4 = self.func(iter(data), scheme)
        self.assertEquals(result1, result2)
        self.assertEquals(result1, result3)
        self.assertEquals(result1, result4)

    def expect_failure(self, data):
        self.assertRaises(ValueError, self.func, data)
        for scheme in self.schemes:
            self.assertRaises(ValueError, self.func, data, scheme)

    # Generic tests that don't care about the specific values:

    def testNoSorting(self):
        """Test that quartiles doesn't sort in place."""
        data = [2, 4, 1, 3, 0, 5]
        save = data[:]
        assert save is not data
        assert data != sorted(data)
        for scheme in self.schemes:
            _ = self.func(data, scheme)
            self.assertEquals(data, save)

    def testTooFewItems(self):
        for data in ([], [1], [1, 2]):
            self.expect_failure(data)

    def testSorted(self):
        # Test that sorted and unsorted data give the same results.
        for n in (8, 9, 10, 11):  # n%4 -> 0...3
            for scheme in self.schemes:
                self.compare_sorted_with_unsorted(n, scheme)

    def testIter(self):
        # Test that iterators and sequences give the same result.
        for n in (8, 9, 10, 11):  # n%4 -> 0...3
            for scheme in self.schemes:
                self.compare_types(n, scheme)

    def testBadScheme(self):
        data = range(20)
        for scheme in ('', 'spam', 1.5, -1.5, -2):
            self.assertRaises(ValueError, self.func, data, scheme)

    def testCaseInsensitive(self):
        data = range(20)
        for scheme in self.func.aliases:
            a = self.func(data, scheme.lower())
            b = self.func(data, scheme.upper())
            c = self.func(data, scheme.title())
            self.assertEquals(a, b)
            self.assertEquals(a, c)

    def testDefaultScheme(self):
        data = list(range(50))
        random.shuffle(data)
        save_scheme = stats.QUARTILE_DEFAULT
        schemes = [1, 2, 3, 4, 5, 6, 'excel', 'minitab']
        try:
            for scheme in schemes:
                stats.QUARTILE_DEFAULT = scheme
                a = stats.quartiles(data)
                b = stats.quartiles(data, scheme)
                self.assertEquals(a, b)
        finally:
            stats.QUARTILE_DEFAULT = save_scheme

    # Tests where we check for the correct result.

    def testInclusive(self):
        # Test the inclusive method of calculating quartiles.
        f = self.func
        scheme = 1
        self.assertEquals(f([0, 1, 2], scheme), (0.5, 1, 1.5))
        self.assertEquals(f([0, 1, 2, 3], scheme), (0.5, 1.5, 2.5))
        self.assertEquals(f([0, 1, 2, 3, 4], scheme), (1, 2, 3))
        self.assertEquals(f([0, 1, 2, 3, 4, 5], scheme), (1, 2.5, 4))
        self.assertEquals(f([0, 1, 2, 3, 4, 5, 6], scheme), (1.5, 3, 4.5))
        self.assertEquals(f(range(1, 9), scheme), (2.5, 4.5, 6.5))
        self.assertEquals(f(range(1, 10), scheme), (3, 5, 7))
        self.assertEquals(f(range(1, 11), scheme), (3, 5.5, 8))
        self.assertEquals(f(range(1, 12), scheme), (3.5, 6, 8.5))
        self.assertEquals(f(range(1, 13), scheme), (3.5, 6.5, 9.5))
        self.assertEquals(f(range(1, 14), scheme), (4, 7, 10))
        self.assertEquals(f(range(1, 15), scheme), (4, 7.5, 11))
        self.assertEquals(f(range(1, 16), scheme), (4.5, 8, 11.5))

    def testExclusive(self):
        # Test the exclusive method of calculating quartiles.
        f = self.func
        scheme = 2
        self.assertEquals(f([0, 1, 2], scheme), (0, 1, 2))
        self.assertEquals(f([0, 1, 2, 3], scheme), (0.5, 1.5, 2.5))
        self.assertEquals(f([0, 1, 2, 3, 4], scheme), (0.5, 2, 3.5))
        self.assertEquals(f([0, 1, 2, 3, 4, 5], scheme), (1, 2.5, 4))
        self.assertEquals(f([0, 1, 2, 3, 4, 5, 6], scheme), (1, 3, 5))
        self.assertEquals(f(range(1, 9), scheme), (2.5, 4.5, 6.5))
        self.assertEquals(f(range(1, 10), scheme), (2.5, 5, 7.5))
        self.assertEquals(f(range(1, 11), scheme), (3, 5.5, 8))
        self.assertEquals(f(range(1, 12), scheme), (3, 6, 9))
        self.assertEquals(f(range(1, 13), scheme), (3.5, 6.5, 9.5))
        self.assertEquals(f(range(1, 14), scheme), (3.5, 7, 10.5))
        self.assertEquals(f(range(1, 15), scheme), (4, 7.5, 11))
        self.assertEquals(f(range(1, 16), scheme), (4, 8, 12))

    def testMS(self):
        f = self.func
        scheme = 3
        self.assertEquals(f(range(3), scheme), (0, 1, 2))
        self.assertEquals(f(range(4), scheme), (0, 1, 3))
        self.assertEquals(f(range(5), scheme), (1, 2, 3))
        self.assertEquals(f(range(6), scheme), (1, 3, 4))
        self.assertEquals(f(range(7), scheme), (1, 3, 5))
        self.assertEquals(f(range(8), scheme), (1, 3, 6))
        self.assertEquals(f(range(9), scheme), (2, 4, 6))
        self.assertEquals(f(range(10), scheme), (2, 5, 7))
        self.assertEquals(f(range(11), scheme), (2, 5, 8))
        self.assertEquals(f(range(12), scheme), (2, 5, 9))

    def testMinitab(self):
        f = self.func
        scheme = 4
        self.assertEquals(f(range(3), scheme), (0, 1, 2))
        self.assertEquals(f(range(4), scheme), (0.25, 1.5, 2.75))
        self.assertEquals(f(range(5), scheme), (0.5, 2, 3.5))
        self.assertEquals(f(range(6), scheme), (0.75, 2.5, 4.25))
        self.assertEquals(f(range(7), scheme), (1, 3, 5))
        self.assertEquals(f(range(8), scheme), (1.25, 3.5, 5.75))
        self.assertEquals(f(range(9), scheme), (1.5, 4, 6.5))
        self.assertEquals(f(range(10), scheme), (1.75, 4.5, 7.25))
        self.assertEquals(f(range(11), scheme), (2, 5, 8))
        self.assertEquals(f(range(12), scheme), (2.25, 5.5, 8.75))

    def testExcel(self):
        f = self.func
        scheme = 5
        # Results generated with OpenOffice.
        self.assertEquals((0.5, 1, 1.5), f(range(3), scheme))
        self.assertEquals((0.75, 1.5, 2.25), f(range(4), scheme))
        self.assertEquals((1, 2, 3), f(range(5), scheme))
        self.assertEquals((1.25, 2.5, 3.75), f(range(6), scheme))
        self.assertEquals((1.5, 3, 4.5), f(range(7), scheme))
        self.assertEquals((1.75, 3.5, 5.25), f(range(8), scheme))
        self.assertEquals((2, 4, 6), f(range(9), scheme))
        self.assertEquals((2.25, 4.5, 6.75), f(range(10), scheme))
        self.assertEquals((2.5, 5, 7.5), f(range(11), scheme))
        self.assertEquals((2.75, 5.5, 8.25), f(range(12), scheme))
        self.assertEquals((3, 6, 9), f(range(13), scheme))
        self.assertEquals((3.25, 6.5, 9.75), f(range(14), scheme))
        self.assertEquals((3.5, 7, 10.5), f(range(15), scheme))

    def testLangford(self):
        f = self.func
        scheme = 6
        self.assertEquals(f(range(3), scheme), (0, 1, 2))
        self.assertEquals(f(range(4), scheme), (0.5, 1.5, 2.5))
        self.assertEquals(f(range(5), scheme), (1, 2, 3))
        self.assertEquals(f(range(6), scheme), (1, 2.5, 4))
        self.assertEquals(f(range(7), scheme), (1, 3, 5))
        self.assertEquals(f(range(8), scheme), (1.5, 3.5, 5.5))
        self.assertEquals(f(range(9), scheme), (2, 4, 6))
        self.assertEquals(f(range(10), scheme), (2, 4.5, 7))
        self.assertEquals(f(range(11), scheme), (2, 5, 8))
        self.assertEquals(f(range(12), scheme), (2.5, 5.5, 8.5))

    def testBig(self):
        data = list(range(1001, 2001))
        assert len(data) == 1000
        assert len(data)%4 == 0
        random.shuffle(data)
        self.assertEquals(self.func(data, 1), (1250.5, 1500.5, 1750.5))
        self.assertEquals(self.func(data, 2), (1250.5, 1500.5, 1750.5))
        data.append(2001)
        random.shuffle(data)
        self.assertEquals(self.func(data, 1), (1251, 1501, 1751))
        self.assertEquals(self.func(data, 2), (1250.5, 1501, 1751.5))
        data.append(2002)
        random.shuffle(data)
        self.assertEquals(self.func(data, 1), (1251, 1501.5, 1752))
        self.assertEquals(self.func(data, 2), (1251, 1501.5, 1752))
        data.append(2003)
        random.shuffle(data)
        self.assertEquals(self.func(data, 1), (1251.5, 1502, 1752.5))
        self.assertEquals(self.func(data, 2), (1251, 1502, 1753))


class HingesTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.func = stats.hinges

    def testNoSorting(self):
        # hinges() does not sort in place.
        data = [2, 4, 1, 3, 0, 5]
        save = data[:]
        assert save is not data
        assert data != sorted(data)
        _ = self.func(data)
        self.assertEquals(data, save)

    def testTooFewItems(self):
        for data in ([], [1], [1, 2]):
            self.assertRaises(ValueError, self.func, data)

    def testSorted(self):
        # Test that sorted and unsorted data give the same results.
        for n in (40, 41, 42, 43):  # n%4 -> 0...3
            data = list(range(n))
            result1 = self.func(data)
            random.shuffle(data)
            result2 = self.func(data)
            self.assertEquals(result1, result2)

    def testTypes(self):
        # Test that iterators and sequences give the same result.
        for n in (40, 41, 42, 43):
            data = range(n)
            result1 = self.func(data)
            data = list(data)
            result2 = self.func(data)
            result3 = self.func(tuple(data))
            result4 = self.func(iter(data))
            self.assertEquals(result1, result2)
            self.assertEquals(result1, result3)
            self.assertEquals(result1, result4)

    def testHinges(self):
        f = self.func
        g = stats.quartiles
        for n in range(3, 25):
            data = range(n)
            self.assertEquals(f(data), g(data, 1))


class QuantileBehaviourTest(unittest.TestCase):
    # Test behaviour of quantile function without caring about
    # the actual values returned.

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.func = stats.quantile

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

    def testDefaultScheme(self):
        data = list(range(50))
        random.shuffle(data)
        save_scheme = stats.QUANTILE_DEFAULT
        schemes = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            'excel', 'minitab',
            (0.375, 0.25, 0, 1),
            ]
        try:
            for scheme in schemes:
                p = random.random()
                stats.QUANTILE_DEFAULT = scheme
                a = stats.quantile(data, p)
                b = stats.quantile(data, p, scheme)
                self.assertEquals(a, b)
        finally:
            stats.QUANTILE_DEFAULT = save_scheme


class QuantileValueTest(unittest.TestCase):
    # Tests of quantile function where we do care about the actual
    # values returned.

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.func = stats.quantile

    def testUnsorted(self):
        data = [3, 4, 2, 1, 0, 5]
        assert data != sorted(data)
        self.assertEquals(self.func(data, 0.1, scheme=1), 0)
        self.assertEquals(self.func(data, 0.9, scheme=1), 5)
        self.assertEquals(self.func(data, 0.1, scheme=7), 0.5)
        self.assertEquals(self.func(data, 0.9, scheme=7), 4.5)

    def testIter(self):
        self.assertEquals(self.func(range(12), 0.3, scheme=1), 3)
        self.assertEquals(self.func(range(12), 0.3, scheme=7), 3.3)

    def testUnitInterval(self):
        data = [0, 1]
        for f in (0.01, 0.1, 0.2, 0.25, 0.5, 0.55, 0.8, 0.9, 0.99):
            self.assertAlmostEquals(self.func(data, f, scheme=7), f, places=9)

    def testLQD(self):
        expected = [1.0, 1.7, 3.9, 6.1, 8.3, 10.5, 12.7, 14.9, 17.1, 19.3, 20.0]
        ps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        data = range(1, 21)
        for i, p in enumerate(ps):
            result = stats.quantile(data, p, scheme=10)
            self.assertAlmostEquals(expected[i], result, places=12)


class QuantilesCompareWithR(unittest.TestCase):
    # Compare results of calling quantile() against results from R.
    places = 3

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.read_data('quantiles.dat')

    def read_data(self, filename):
        # Read data from external test data file generated using R.
        expected = {}
        with open(filename, 'r') as data:
            for line in data:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if '=' in line:
                    label, items = line.split("=")
                    label = label.strip()
                    if label == 'seq':
                        start, end, step = [int(s) for s in items.split()]
                        end += 1
                        self.data = list(range(start, end, step))
                    elif label == 'p':
                        self.fractiles = [float(s) for s in items.split()]
                else:
                    scheme, results = line.split(":")
                    scheme = int(scheme.strip())
                    assert 1 <= scheme <= 9
                    results = [float(x) for x in results.split()]
                    expected[scheme] = results
        self.expected = expected

    def compare(self, scheme):
        a = [round(stats.quantile(self.data, p, scheme=scheme), self.places)
             for p in self.fractiles]
        b = [round(x, self.places) for x in self.expected[scheme]]
        self.assertEquals(a, b)

    def testR1(self):  self.compare(1)
    def testR2(self):  self.compare(2)
    def testR3(self):  self.compare(3)
    def testR4(self):  self.compare(4)
    def testR5(self):  self.compare(5)
    def testR6(self):  self.compare(6)
    def testR7(self):  self.compare(7)
    def testR8(self):  self.compare(8)
    def testR9(self):  self.compare(9)


class CompareQuantileMethods(unittest.TestCase):
    data1 = list(range(1000, 2000, 100))
    data2 = list(range(2000, 3001, 100))
    assert len(data1)%2 == 0
    assert len(data2)%2 == 1

    fractions = [0.0, 0.01, 0.1, 0.2, 0.25, 0.31, 0.42, 0.5, 0.55,
                0.62, 0.75, 0.83, 0.9, 0.95, 0.99, 1.0]

    def compareMethods(self, scheme, params):
        for p in self.fractions:
            a = stats.quantile(self.data1, p, scheme=scheme)
            b = stats.quantile(self.data1, p, scheme=params)
            self.assertEquals(a, b, "%s != %s; p=%f" % (a, b, p))
            a = stats.quantile(self.data2, p, scheme=scheme)
            b = stats.quantile(self.data2, p, scheme=params)
            self.assertEquals(a, b, "%s != %s; p=%f" % (a, b, p))

    def testR1(self):
        scheme = 1; params = (0, 0, 1, 0)
        self.compareMethods(scheme, params)

    # Note that there is *no* test for R2, as it is not supported by the
    # Mathematica parameterized quantile algorithm.

    @unittest.skip('test currently broken for unknown reasons')
    def testR3(self):
        scheme = 3; params = (0.5, 0, 0, 0)
        self.compareMethods(scheme, params)

    def testR4(self):
        scheme = 4; params = (0, 0, 0, 1)
        self.compareMethods(scheme, params)

    def testR5(self):
        scheme = 5; params = (0.5, 0, 0, 1)
        self.compareMethods(scheme, params)

    def testR6(self):
        scheme = 6; params = (0, 1, 0, 1)
        self.compareMethods(scheme, params)

    def testR7(self):
        scheme = 7; params = (1, -1, 0, 1)
        self.compareMethods(scheme, params)

    def testR8(self):
        scheme = 8; params = (1/3, 1/3, 0, 1)
        self.compareMethods(scheme, params)

    def testR9(self):
        scheme = 9; params = (3/8, 0.25, 0, 1)
        self.compareMethods(scheme, params)


class DecileTest(unittest.TestCase):
    def testSimple(self):
        data = range(1, 11)
        for i in range(1, 11):
            self.assertEquals(stats.decile(data, i, scheme=1), i)


class PercentileTest(unittest.TestCase):
    def testSimple(self):
        data = range(1, 101)
        for i in range(1, 101):
            self.assertEquals(stats.percentile(data, i, scheme=1), i)


class BoxWhiskerPlotTest(unittest.TestCase):
    pass


# Test spread statistics
# ----------------------

class TinyVariance(unittest.TestCase):
    # Minimal tests for variance and friends.
   def testVariance(self):
       data = [1, 2, 3]
       assert stats.mean(data) == 2
       self.assertEquals(stats.pvariance(data), 2/3)
       self.assertEquals(stats.variance(data), 1.0)
       self.assertEquals(stats.pvariance1(data), 2/3)
       self.assertEquals(stats.variance1(data), 1.0)
       self.assertEquals(stats.pstdev(data), math.sqrt(2/3))
       self.assertEquals(stats.stdev(data), 1.0)
       self.assertEquals(stats.pstdev1(data), math.sqrt(2/3))
       self.assertEquals(stats.stdev1(data), 1.0)


class PVarianceTest(unittest.TestCase):
    # General test data:
    func = stats.pvariance
    data = (4.0, 7.0, 13.0, 16.0)
    expected = 22.5  # Exact population variance.
    # Test data for exact (uniform distribution) test:
    uniform_data = range(10000)
    uniform_expected = (10000**2 - 1)/12
    # Expected result calculated by HP-48GX:
    hp_expected = 88349.2408884
    # Scaling factor when you duplicate each data point:
    scale = 1.0

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

    def test_uniform(self):
        # Compare the calculated variance against an exact result.
        self.assertEquals(self.func(self.uniform_data), self.uniform_expected)

    def testCompareHP(self):
        # Compare against a result calculated with a HP-48GX calculator.
        data = (list(range(1, 11)) + list(range(1000, 1201)) +
            [0, 3, 7, 23, 42, 101, 111, 500, 567])
        random.shuffle(data)
        self.assertAlmostEquals(self.func(data), self.hp_expected, places=7)

    def testDuplicate(self):
        data = [random.uniform(-100, 500) for _ in range(20)]
        a = self.func(data)
        b = self.func(data*2)
        self.assertAlmostEquals(a*self.scale, b, places=9)


class VarianceTest(PVarianceTest):
    func = stats.variance
    expected = 30.0  # Exact sample variance.
    uniform_expected = PVarianceTest.uniform_expected * 10000/(10000-1)
    hp_expected = 88752.6620797
    scale = (2*20-2)/(2*20-1)

    def testSingletonFailure(self):
        for data in ([1], iter([1])):
            self.assertRaises(ValueError, self.func, data)

    def testCompareR(self):
        # Compare against a result calculated with R code:
        #   > x <- c(seq(1, 10), seq(1000, 1200))
        #   > var(x)
        #   [1] 57563.55
        data = list(range(1, 11)) + list(range(1000, 1201))
        expected = 57563.55
        self.assertAlmostEquals(self.func(data), expected, places=3)
        # The expected value from R looks awfully precise... are they
        # rounding it, or is that the exact value?
        # My HP-48GX calculator returns 57563.5502144.


class PStdevTest(PVarianceTest):
    func = stats.pstdev
    expected = math.sqrt(22.5)  # Exact population stdev.
    uniform_expected = math.sqrt(PVarianceTest.uniform_expected)
    hp_expected = 297.236002006


class StdevTest(VarianceTest):
    func = stats.stdev
    expected = math.sqrt(30.0)  # Exact sample stdev.
    uniform_expected = math.sqrt(VarianceTest.uniform_expected)
    hp_expected = 297.913850097
    scale = math.sqrt(VarianceTest.scale)

    def testCompareR(self):
        data = list(range(1, 11)) + list(range(1000, 1201))
        expected = 239.9241
        self.assertAlmostEquals(self.func(data), expected, places=4)


class PVariance1Test(PVarianceTest):
    func = stats.pvariance1


class Variance1Test(VarianceTest):
    func = stats.variance1


class PStdev1Test(PStdevTest):
    func = stats.pstdev1


class Stdev1Test(StdevTest):
    func = stats.stdev1


class VarianceMeanTest(unittest.TestCase):

    def compare_with_and_without_mean(self, func):
        mu = 100*random.random()
        sigma = 10*random.random()+1
        for data in (
            [-6, -2, 0, 3, 4, 5, 5, 5, 6, 7, 9, 11, 15, 25, 26, 27, 28, 42],
            [random.random() for _ in range(10)],
            [random.uniform(10000, 11000) for _ in range(50)],
            [random.uniform(mu, sigma) for _ in range(50)],
            ):
            m = stats.mean(data)
            a = func(data)
            b = func(data, m)
            self.assertEquals(a, b)

    def test_pvar(self):
        self.compare_with_and_without_mean(stats.pvariance)

    def test_var(self):
        self.compare_with_and_without_mean(stats.variance)

    def test_pstdev(self):
        self.compare_with_and_without_mean(stats.pstdev)

    def test_stdev(self):
        self.compare_with_and_without_mean(stats.stdev)


class RangeTest(unittest.TestCase):
    def testFailure(self):
        self.assertRaises(ValueError, stats.range, [])
        self.assertRaises(ValueError, stats.range, iter([]))

    def testSingleton(self):
        for x in (-3.1, 0.0, 4.2, 1.789e12):
            self.assertEquals(stats.range([x]), 0)

    def generate_data_sets(self):
        """Yield 2-tuples of (data, expected range)."""
        # List arguments:
        yield ([42], 0)
        yield ([1, 5], 4)
        yield ([1.5, 4.5, 9.0], 7.5)
        yield ([5, 5, 5], 0)
        data = list(range(500))
        random.shuffle(data)
        for shift in (0, 0.5, 1234.567, -1000, 1e6, 1e9):
            d = [x + shift for x in data]
            yield (d, 499)
        # Subclass of list:
        class MyList(list):
            pass
        yield (MyList([1, 2, 3, 4, 5, 6]), 5)
        yield (MyList([-1, 0, 1, 2, 3, 4, 5]), 6)
        yield (MyList([-1, -2, -3, -4, -5]), 4)
        # Tuple arguments:
        yield ((7, 2), 5)
        yield ((7, 2, 5, 6), 5)
        yield ((3.25, 7.5, 3.25, 4.2), 4.25)
        # range objects:
        yield (range(11), 10)
        yield (range(11, -1, -1), 11)

    def testSequence(self):
        for data, expected in self.generate_data_sets():
            self.assertEquals(stats.range(data), expected)

    def testIterator(self):
        for data, expected in self.generate_data_sets():
            self.assertEquals(stats.range(iter(data)), expected)


class IQRTest(unittest.TestCase):

    def testBadSelector(self):
        for method in (-1, 1.5, "spam"):
            self.assertRaises(ValueError, stats.iqr, [1, 2, 3, 4], method)

    def testBadData(self):
        for data in ([], [1], [2, 3]):
            self.assertRaises(ValueError, stats.iqr, data)

    def testCaseInsensitive(self):
        data = [1, 2, 3, 6, 9, 12, 18, 22]
        for name, num in stats.quartiles.aliases.items():
            a = stats.iqr(data, name.lower())
            b = stats.iqr(data, name.upper())
            c = stats.iqr(data, num)
            self.assertEqual(a, b)
            self.assertEqual(a, c)


class AverageDeviationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.func = stats.average_deviation

    def testTooFewItems(self):
        self.assertRaises(ValueError, self.func, [])

    def testCompareSorting(self):
        # Ensure results don't depend whether input is sorted or not.
        for data in (range(23), range(42, 84, 7), range(-10, 10, 3)):
            data = sorted(data)
            result1 = self.func(data)
            random.shuffle(data)
            result2 = self.func(data)
            self.assertEquals(result1, result2)

    def testCompareTypes(self):
        # Ensure results don't depend on the type of the input.
        for data in (range(20), range(30, 70, 7), range(-10, 10, 3)):
            result1 = self.func(data)
            data = list(data)
            result2 = self.func(data)
            data = tuple(data)
            result3 = self.func(data)
            data = iter(data)
            result4 = self.func(data)
            self.assertEquals(result1, result2)
            self.assertEquals(result1, result3)
            self.assertEquals(result1, result4)

    def testSuppliedMean(self):
        # Test that pre-calculating the mean gives the same result.
        for data in (range(35), range(-17, 53, 7), range(11, 79, 3)):
            data = list(data)
            random.shuffle(data)
            m = stats.mean(data)
            result1 = self.func(data)
            result2 = self.func(data, m)
            self.assertEquals(result1, result2)

    def testSingleton(self):
        self.assertEquals(self.func([42]), 0)
        self.assertEquals(self.func([42], 40), 2)

    def testMain(self):
        data = [-1.25, 0.5, 0.5, 1.75, 3.25, 4.5, 4.5, 6.25, 6.75, 9.75]
        expected = 2.7
        for delta in (0, 100, 1e6, 1e9):
            self.assertEquals(self.func(x+delta for x in data), expected)


class MedianAverageDeviationTest(AverageDeviationTest):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.func = stats.median_average_deviation

    def testSuppliedMedian(self):
        # Test that pre-calculating the median gives the same result.
        for data in (range(35), range(-17, 53, 7), range(11, 79, 3)):
            result1 = self.func(data)
            m = stats.median(data)
            data = list(data)
            random.shuffle(data)
            result2 = self.func(data, m)
            self.assertEquals(result1, result2)

    def testMain(self):
        data = [-1.25, 0.5, 0.5, 1.75, 3.25, 4.5, 4.5, 6.25, 6.75, 9.75]
        expected = 2.625
        for delta in (0, 100, 1e6, 1e9):
            self.assertEquals(self.func(x+delta for x in data), expected)

    def testNoScaling(self):
        # Test alternative ways of spelling no scaling factor.
        data = [random.random()+23 for _ in range(100)]
        expected = self.func(data)
        for scale in (1, None, 'none'):
            self.assertEquals(self.func(data, scale=scale), expected)

    def testScales(self):
        data = [100*random.random()+42 for _ in range(100)]
        expected = self.func(data)
        self.assertEquals(self.func(data, scale='normal'), expected*1.4826)
        self.assertAlmostEquals(self.func(data, scale='uniform'),
            expected*1.1547, places=4) # Documented value in docstring.
        self.assertEquals(self.func(data, scale='uniform'),
            expected*math.sqrt(4/3))  # Exact value.
        for x in (-1.25, 0.0, 1.25, 4.5, 9.75):
            self.assertEquals(self.func(data, scale=x), expected*x)

    def testCaseInsensitive(self):
        for scale in ('normal', 'uniform', 'none'):
            data = [67*random.random()+19 for _ in range(100)]
            a = self.func(data, scale=scale.lower())
            b = self.func(data, scale=scale.upper())
            c = self.func(data, scale=scale.title())
            self.assertEquals(a, b)
            self.assertEquals(a, c)

    def testHasScaling(self):
        self.assert_(hasattr(self.func, 'scaling'))

    def testSignOdd(self):
        data = [23*random.random()+42 for _ in range(55)]
        assert len(data)%2 == 1
        a = self.func(data, sign=-1)
        b = self.func(data, sign=0)
        c = self.func(data, sign=1)
        self.assertEquals(a, b)
        self.assertEquals(a, c)

    def testSignEven(self):
        data = [0.5, 1.5, 3.25, 4.25, 6.25, 6.75]
        assert len(data)%2 == 0
        self.assertEquals(self.func(data, sign=-1), 1.75)
        self.assertEquals(self.func(data, sign=0), 2.375)
        self.assertEquals(self.func(data), 2.375)
        self.assertEquals(self.func(data, sign=1), 2.5)


# Test other moments
# ------------------

class QuartileSkewnessTest(unittest.TestCase):
    def testFailure(self):
        self.assertRaises(ValueError, stats.quartile_skewness, 2, 3, 1)

    def testNan(self):
        x = stats.quartile_skewness(1, 1, 1)
        self.assert_(math.isnan(x))

    def testSkew(self):
        self.assertEquals(stats.quartile_skewness(3, 5, 7), 0.0)
        self.assertEquals(stats.quartile_skewness(0, 1, 10), 0.8)
        self.assertEquals(stats.quartile_skewness(0, 9, 10), -0.8)


class PearsonModeSkewnessTest(unittest.TestCase):
    def testFailure(self):
        self.assertRaises(ValueError, stats.pearson_mode_skewness, 2, 3, -1)

    def testNan(self):
        x = stats.pearson_mode_skewness(5, 5, 0)
        self.assert_(math.isnan(x))

    def testInf(self):
        x = stats.pearson_mode_skewness(3, 2, 0)
        self.assert_(math.isinf(x))

    def testSkew(self):
        self.assertEquals(stats.pearson_mode_skewness(2.5, 2.25, 2.5), 0.1)


class SkewnessTest(unittest.TestCase):
    def test_uniform(self):
        # Compare the calculated skewness against an exact result
        # calculated from a uniform distribution.
        data = range(10000)
        self.assertEquals(stats.skewness(data), 0.0)
        data = [x + 1e9 for x in data]
        self.assertEquals(stats.skewness(data), 0.0)

    def test_shift0(self):
        data = [(2*i+1)/4 for i in range(1000)]
        random.shuffle(data)
        k1 = stats.skewness(data)
        self.assertEquals(k1, 0.0)
        k2 = stats.skewness(x+1e9 for x in data)
        self.assertEquals(k2, 0.0)

    def test_shift(self):
        d1 = [(2*i+1)/3 for i in range(1000)]
        d2 = [(3*i-19)/2 for i in range(1000)]
        data = [x*y for x,y in zip(d1, d2)]
        random.shuffle(data)
        k1 = stats.skewness(data)
        k2 = stats.skewness(x+1e9 for x in data)
        self.assertAlmostEquals(k1, k2, places=7)

    def test_types(self):
        # Results should be the same no matter what type is used.
        d1 = [(3*i+1)/4 for i in range(1000)]
        d2 = [(2*i+3)/5 for i in range(1000)]
        random.shuffle(d1)
        random.shuffle(d2)
        data = [x*y for x,y in zip(d1, d2)]
        a = stats.skewness(data)
        b = stats.skewness(tuple(data))
        c = stats.skewness(iter(data))
        self.assertEquals(a, b)
        self.assertEquals(a, c)

    def test_sorted(self):
        # Results should not depend on whether data is sorted or not.
        d1 = [(9*i-11)/5 for i in range(100)]
        d2 = [(7*i+2)/7 for i in range(100)]
        random.shuffle(d1)
        random.shuffle(d2)
        data = [x*y for x,y in zip(d1, d2)]
        a = stats.skewness(data)
        b = stats.skewness(sorted(data))
        self.assertAlmostEquals(a, b, places=14)

    def testMeanStdev(self):
        # Giving the sample mean and/or stdev shouldn't change the result.
        d1 = [(98-3*i)/6 for i in range(100)]
        d2 = [(14*i-3)/2 for i in range(100)]
        random.shuffle(d1)
        random.shuffle(d2)
        data = [x*y for x,y in zip(d1, d2)]
        m = stats.mean(data)
        s = stats.stdev(data)
        a = stats.skewness(data)
        b = stats.skewness(data, m)
        c = stats.skewness(data, None, s)
        d = stats.skewness(data, m, s)
        self.assertEquals(a, b)
        self.assertEquals(a, c)
        self.assertEquals(a, d)


class KurtosisTest(unittest.TestCase):

    def corrected_uniform_kurtosis(self, n):
        """Return the exact kurtosis for a discrete uniform distribution."""
        # Calculate the exact population kurtosis:
        expected = -6*(n**2 + 1)/(5*(n - 1)*(n + 1))
        # Give a correction factor to adjust it for sample kurtosis:
        expected *= (n/(n-1))**3
        # XXX why 3? I can't justify this, but it seems to give the closest
        # results.
        return expected

    def test_uniform(self):
        # Compare the calculated kurtosis against an exact result
        # calculated from a uniform distribution.
        n = 10000
        data = range(n)
        expected = self.corrected_uniform_kurtosis(n)
        self.assertAlmostEquals(stats.kurtosis(data), expected, places=6)
        data = [x + 1e9 for x in data]
        self.assertAlmostEquals(stats.kurtosis(data), expected, places=6)

    def test_shift0(self):
        data = [(2*i+1)/4 for i in range(1000)]
        random.shuffle(data)
        k1 = stats.kurtosis(data)
        k2 = stats.kurtosis(x+1e9 for x in data)
        self.assertEquals(k1, k2)

    def test_shift(self):
        d1 = [(2*i+1)/3 for i in range(1000)]
        d2 = [(3*i-19)/2 for i in range(1000)]
        random.shuffle(d1)
        random.shuffle(d2)
        data = [x*y for x,y in zip(d1, d2)]
        k1 = stats.kurtosis(data)
        k2 = stats.kurtosis(x+1e9 for x in data)
        self.assertAlmostEquals(k1, k2, places=9)

    def test_types(self):
        # Results should be the same no matter what type is used.
        d1 = [(3*i-19)/8 for i in range(100)]
        d2 = [(12*i+5)/11 for i in range(100)]
        random.shuffle(d1)
        random.shuffle(d2)
        data = [x*y for x,y in zip(d1, d2)]
        a = stats.kurtosis(data)
        b = stats.kurtosis(tuple(data))
        c = stats.kurtosis(iter(data))
        self.assertEquals(a, b)
        self.assertEquals(a, c)

    def test_sorted(self):
        # Results should not depend on whether data is sorted or not.
        d1 = [(15*i+1)/3 for i in range(100)]
        d2 = [(4*i-26)/5 for i in range(100)]
        random.shuffle(d1)
        random.shuffle(d2)
        data = [x*y for x,y in zip(d1, d2)]
        a = stats.kurtosis(data)
        b = stats.kurtosis(sorted(data))
        self.assertAlmostEquals(a, b, places=14)

    def testMeanStdev(self):
        # Giving the sample mean and/or stdev shouldn't change the result.
        d1 = [(17*i-45)/16 for i in range(100)]
        d2 = [(9*i-25)/3 for i in range(100)]
        random.shuffle(d1)
        random.shuffle(d2)
        data = [x*y for x,y in zip(d1, d2)]
        m = stats.mean(data)
        s = stats.stdev(data)
        a = stats.kurtosis(data)
        b = stats.kurtosis(data, m)
        c = stats.kurtosis(data, None, s)
        d = stats.kurtosis(data, m, s)
        self.assertEquals(a, b)
        self.assertEquals(a, c)
        self.assertEquals(a, d)


# Test multivariate statistics
# ----------------------------

class QCorrTest(unittest.TestCase):
    def testPerfectCorrelation(self):
        xdata = range(-42, 1100, 7)
        ydata = [3.5*x - 0.1 for x in xdata]
        self.assertEquals(stats.qcorr(xdata, ydata), 1.0)

    def testPerfectAntiCorrelation(self):
        xydata = [(1, 10), (2, 8), (3, 6), (4, 4), (5, 2)]
        self.assertEquals(stats.qcorr(xydata), -1.0)
        xdata = range(-23, 1000, 3)
        ydata = [875.1 - 4.2*x for x in xdata]
        self.assertEquals(stats.qcorr(xdata, ydata), -1.0)

    def testPerfectZeroCorrelation(self):
        data = []
        for x in range(1, 10):
            for y in range(1, 10):
                data.append((x, y))
        random.shuffle(data)
        self.assertEquals(stats.qcorr(data), 0)

    def testCompareAlternateInput(self):
        # Compare the xydata vs. xdata, ydata input arguments.
        xdata = [random.random() for _ in range(1000)]
        ydata = [random.random() for _ in range(1000)]
        a = stats.qcorr(xdata, ydata)
        b = stats.qcorr(list(zip(xdata, ydata)))
        self.assertEquals(a, b)

    def testNan(self):
        # Vertical line:
        xdata = [1 for _ in range(50)]
        ydata = [random.random() for _ in range(50)]
        result = stats.qcorr(xdata, ydata)
        self.assert_(math.isnan(result))
        # Horizontal line:
        xdata = [random.random() for _ in range(50)]
        ydata = [1 for _ in range(50)]
        result = stats.qcorr(xdata, ydata)
        self.assert_(math.isnan(result))
        # Neither horizontal nor vertical:
        # Take x-values and y-values both = (1, 2, 2, 3) with median = 2.
        xydata = [(1, 2), (2, 3), (2, 1), (3, 2)]
        result = stats.qcorr(xydata)
        self.assert_(math.isnan(result))

    def testEmpty(self):
        self.assertRaises(ValueError, stats.qcorr, [])
        self.assertRaises(ValueError, stats.qcorr, [], [])

    def testTypes(self):
        xdata = [random.random() for _ in range(20)]
        ydata = [random.random() for _ in range(20)]
        a = stats.qcorr(xdata, ydata)
        b = stats.qcorr(tuple(xdata), tuple(ydata))
        c = stats.qcorr(iter(xdata), iter(ydata))
        d = stats.qcorr(zip(xdata, ydata))
        self.assertEquals(a, b)
        self.assertEquals(a, c)
        self.assertEquals(a, d)


class CorrTest(unittest.TestCase):
    # Common tests for corr() and corr1().
    # All calls to the test function must be the one-argument style.
    # See CorrExtrasTest for two-argument tests.

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.func = stats.corr

    def testOrdered(self):
        # Order shouldn't matter.
        xydata = [(x, 2.7*x - 0.3) for x in range(-20, 30)]
        a = self.func(xydata)
        random.shuffle(xydata)
        b = self.func(xydata)
        self.assertEquals(a, b)

    def testPerfectCorrelation(self):
        xydata = [(x, 2.3*x - 0.8) for x in range(-17, 395, 3)]
        self.assertEquals(self.func(xydata), 1.0)

    def testPerfectAntiCorrelation(self):
        xydata = [(x, 273.4 - 3.1*x) for x in range(-22, 654, 7)]
        self.assertEquals(self.func(xydata), -1.0)

    def testPerfectZeroCorrelation(self):
        data = []
        for x in range(1, 10):
            for y in range(1, 10):
                data.append((x, y))
        self.assertEquals(self.func(data), 0)

    def testFailures(self):
        # One argument version.
        self.assertRaises(ValueError, self.func, [])
        self.assertRaises(ValueError, self.func, [(1, 3)])

    def testTypes(self):
        # The type of iterable shouldn't matter.
        xdata = [random.random() for _ in range(20)]
        ydata = [random.random() for _ in range(20)]
        xydata = zip(xdata, ydata)
        a = self.func(xydata)
        xydata = list(zip(xdata, ydata))
        b = self.func(xydata)
        c = self.func(tuple(xydata))
        d = self.func(iter(xydata))
        self.assertEquals(a, b)
        self.assertEquals(a, c)
        self.assertEquals(a, d)

    def testExact(self):
        xdata = [0, 10, 4, 8, 8]
        ydata = [2, 6, 2, 4, 6]
        self.assertEquals(self.func(zip(xdata, ydata)), 28/32)

    def testHP_1(self):
        # Compare to results calculated on HP-48GX using this function:
        # << CL-SIGMA -5 15 FOR X X 2 X - X SQ + ->V2 SIGMA+ .1 STEP >>
        xdata = [i/10 for i in range(-50, 151)]
        ydata = [x**2 - x + 2 for x in xdata]
        assert len(xdata) == len(ydata) == 201
        self.assertAlmostEquals(sum(xdata), 1005, places=12)
        self.assertAlmostEquals(sum(ydata), 11189, places=11)
        expected = 0.866300845681
        result = self.func(zip(xdata, ydata))
        self.assertAlmostEquals(result, expected, places=12)
        # For future use: COV = 304.515, PCOV = 303 LINFIT = 10.666...67 + 9x

    def testHP_2(self):
        # Compare to results calculated on HP-48GX using this function:
        # << CL-SIGMA -30 60 FOR I I 3 / 500 I + SQRT ->V2 SIGMA+ NEXT >>
        xdata = [i/3 for i in range(-30, 61)]
        ydata = [math.sqrt(500 + i) for i in range(-30, 61)]
        assert len(xdata) == len(ydata) == 91
        self.assertAlmostEquals(sum(xdata), 455, places=12)
        self.assertAlmostEquals(sum(ydata), 2064.4460877, places=6)
        expected = 0.999934761605
        result = self.func(zip(xdata, ydata))
        self.assertAlmostEquals(result, expected, places=12)
        # For future use: COV = 5.1268171707, PCOV = 5.07047852047
        # LINFIT = 22.3555373622 + 6.61366763539e-2x

    def testDuplicate(self):
        # corr shouldn't change if you duplicate each point.
        # Try first with a high correlation.
        xdata = [random.uniform(-5, 15) for _ in range(15)]
        ydata = [x - 0.5 + random.random() for x in xdata]
        a = self.func(zip(xdata, ydata))
        b = self.func(zip(xdata*2, ydata*2))
        self.assertAlmostEquals(a, b, places=12)
        # And again with a (probably) low correlation.
        ydata = [random.uniform(-5, 15) for _ in range(15)]
        a = self.func(zip(xdata, ydata))
        b = self.func(zip(xdata*2, ydata*2))
        self.assertAlmostEquals(a, b, places=12)

    def testSame(self):
        data = [random.random() for x in range(5)]
        result = self.func([(x, x) for x in data])
        self.assertAlmostEquals(result, 1.0, places=14)  # small list
        data = [random.random() for x in range(100)]
        result = self.func([(x, x) for x in data])
        self.assertAlmostEquals(result, 1.0, places=14)  # medium list
        data = [random.random() for x in range(100000)]
        result = self.func([(x, x) for x in data])
        self.assertAlmostEquals(result, 1.0, places=14)  # large list

    def generate_stress_data(self, start, end, step):
        xfuncs = (lambda x: x, lambda x: 12345*x + 9876,
                  lambda x: 1e9*x, lambda x: 1e-9*x)
        yfuncs = (lambda y: y, lambda y: 67890*y + 6428,
                  lambda y: 1e9*y, lambda y: 1e-9*y)
        for i in range(start, end, step):
            xdata = [random.random() for _ in range(i)]
            ydata = [random.random() for _ in range(i)]
            for fx, fy in [(fx,fy) for fx in xfuncs for fy in yfuncs]:
                xs = [fx(x) for x in xdata]
                ys = [fy(y) for y in ydata]
                yield (xs, ys)

    def testStress(self):
        # Stress the corr() function looking for failures of the
        # post-condition -1 <= r <= 1.
        for xdata, ydata in self.generate_stress_data(5, 51, 3):
            result = self.func(zip(xdata, ydata))
            self.assertTrue(-1.0 <= result <= 1.0)

    def testShift(self):
        xdata = [random.random() for _ in range(50)]
        ydata = [random.random() for _ in range(50)]
        a = self.func(zip(xdata, ydata))
        for x0, y0 in [(42, -99), (1.2e6, 4.5e5), (7.8e9, 3.6e9)]:
            xdata = [x+x0 for x in xdata]
            ydata = [y+y0 for y in ydata]
            b = self.func(zip(xdata, ydata))
            self.assertAlmostEquals(a, b, places=5)  # FIXME


class CorrExtrasTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.func = stats.corr

    def testSimple(self):
        # Simple test originally from a doctest in _corr2.
        xdata = [0.0, 0.1, 0.25, 1.2, 1.75]
        ydata = [2.5*x + 0.3 for x in xdata]  # Perfect correlation.
        self.assertAlmostEquals(self.func(xdata, ydata), 1.0, places=14)
        ydata = [10-y for y in ydata]
        self.assertAlmostEquals(self.func(xdata, ydata), -1.0, places=14)

    def testCompareAlternateInput(self):
        # Compare the xydata vs. xdata, ydata input arguments.
        xdata = [random.random() for _ in range(1000)]
        ydata = [random.random() for _ in range(1000)]
        a = self.func(xdata, ydata)
        b = self.func(zip(xdata, ydata))
        self.assertEquals(a, b)

    def testFailures(self):
        # Two argument version.
        self.assertRaises(ValueError, self.func, [], [])
        self.assertRaises(ValueError, self.func, [12], [46])

    def testTypes(self):
        # The type of iterable shouldn't matter.
        xdata = [random.random() for _ in range(20)]
        ydata = [random.random() for _ in range(20)]
        a = self.func(xdata, ydata)
        b = self.func(tuple(xdata), tuple(ydata))
        c = self.func(iter(xdata), iter(ydata))
        self.assertEquals(a, b)
        self.assertEquals(a, c)

    def stress_test(self, xdata, ydata):
        xfuncs = (lambda x: -1.2345e7*x - 23.42, lambda x: 9.42e-6*x + 2.1)
        yfuncs = (lambda y: -2.9234e7*y + 1.97, lambda y: 7.82e8*y - 307.9)
        for fx, fy in [(fx,fy) for fx in xfuncs for fy in yfuncs]:
            xs = [fx(x) for x in xdata]
            ys = [fy(y) for y in ydata]
            result = self.func(xs, ys)
            self.assertTrue(-1.0 <= result <= 1.0)

    def testStress(self):
        # A few extra stress tests.
        for i in range(6, 22, 3):
            xdata = [random.uniform(-100, 300) for _ in range(i)]
            ydata = [random.uniform(-5000, 5000) for _ in range(i)]
            self.stress_test(xdata, ydata)


class Corr1Test(CorrTest):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.func = stats.corr1

    def testPerfectCorrelation(self):
        xydata = [(x, 2.3*x - 0.8) for x in range(-17, 395, 3)]
        self.assertAlmostEquals(self.func(xydata), 1.0, places=14)

    def testPerfectZeroCorrelation(self):
        data = []
        for x in range(1, 10):
            for y in range(1, 10):
                data.append((x, y))
        self.assertAlmostEquals(self.func(data), 0.0, places=14)

    def testOrdered(self):
        # Order shouldn't matter.
        xydata = [(x, 2.7*x - 0.3) for x in range(-20, 30)]
        a = self.func(xydata)
        random.shuffle(xydata)
        b = self.func(xydata)
        self.assertAlmostEquals(a, b, places=15)

    def testStress(self):
        # Stress the corr1() function looking for failures of the
        # post-condition -1 <= r <= 1. Don't stop on the first error.
        failed = 0
        it = self.generate_stress_data(5, 51, 1)
        for count, (xdata, ydata) in enumerate(it, 1):
            result = self.func(zip(xdata, ydata))
            failed += not -1.0 <= result <= 1.0
        assert count == 736
        self.assertEquals(failed, 0,
            "%d out of %d out of range errors" % (failed, count))


class PCovTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.func = stats.pcov

    def testEmpty(self):
        self.assertRaises(ValueError, self.func, [])

    def testSingleton(self):
        self.assertEquals(self.func([(1, 2)]), 0.0)

    def testSymmetry(self):
        data1 = [random.random() for _ in range(10)]
        data2 = [random.random() for _ in range(10)]
        a = self.func(zip(data1, data2))
        b = self.func(zip(data2, data1))
        self.assertEquals(a, b)

    def testEqualPoints(self):
        # Equal X values.
        data = [(23, random.random()) for _ in range(50)]
        self.assertEquals(self.func(data), 0.0)
        # Equal Y values.
        data = [(random.random(), 42) for _ in range(50)]
        self.assertEquals(self.func(data), 0.0)
        # Both equal.
        data = [(23, 42)]*50
        self.assertEquals(self.func(data), 0.0)

    def testReduce(self):
        # Covariance reduces to variance if X == Y.
        data = [random.random() for _ in range(50)]
        a = stats.pvariance(data)
        b = self.func(zip(data, data))
        self.assertEquals(a, b)

    def testShift(self):
        xdata = [random.random() for _ in range(50)]
        ydata = [random.random() for _ in range(50)]
        a = self.func(zip(xdata, ydata))
        for x0, y0 in [(-23, 89), (3.7e5, 2.9e6), (1.4e9, 8.1e9)]:
            xdata = [x+x0 for x in xdata]
            ydata = [y+y0 for y in ydata]
            b = self.func(zip(xdata, ydata))
            self.assertAlmostEquals(a, b, places=6)


class CovTest(PCovTest):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.func = stats.cov

    def testSingleton(self):
        self.assertRaises(ValueError, self.func, [(1, 2)])

    def testReduce(self):
        # Covariance reduces to variance if X == Y.
        data = [random.random() for _ in range(50)]
        a = stats.variance(data)
        b = self.func(zip(data, data))
        self.assertEquals(a, b)


class ErrSumSqTest(unittest.TestCase):
    pass


class LinrTest(unittest.TestCase):
    pass


# Test sums and products
# ----------------------

class SumTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.func = stats.sum

    def testEmpty(self):
        self.assertEquals(self.func([]), 0)
        self.assertEquals(self.func([], 123.456), 123.456)

    def testSorted(self):
        # Sum shouldn't depend on the order of items.
        data = [i/7 for i in range(-35, 36)]
        a = self.func(data)
        random.shuffle(data)
        b = self.func(data)
        self.assertEquals(a, b)

    def testSum(self):
        data = [random.random() for _ in range(100)]
        self.assertEquals(self.func(data), math.fsum(data))

    def testTypes(self):
        for data in (range(23), range(-35, 36), range(-23, 42, 7)):
            a = self.func(data)
            b = self.func(list(data))
            c = self.func(tuple(data))
            d = self.func(iter(data))
            self.assertEquals(a, b)
            self.assertEquals(a, c)
            self.assertEquals(a, d)

    def testExact(self):
        # sum of 1, 2, 3, ... n = n(n+1)/2
        data = range(1, 131)
        expected = 130*131/2
        self.assertEquals(self.func(data), expected)
        # sum of squares of 1, 2, 3, ... n = n(n+1)(2n+1)/6
        data = [n**2 for n in range(1, 57)]
        expected = 56*57*(2*56+1)/6
        self.assertEquals(self.func(data), expected)
        # sum of cubes of 1, 2, 3, ... n = n**2(n+1)**2/4 = (1+2+...+n)**2
        data1 = range(1, 85)
        data2 = [n**3 for n in data1]
        expected = (84**2*85**2)/4
        self.assertEquals(self.func(data1)**2, expected)
        self.assertEquals(self.func(data2), expected)

    def testStart(self):
        data = [random.uniform(1, 1000) for _ in range(100)]
        t = self.func(data)
        self.assertEquals(t+42, self.func(data, 42))
        self.assertEquals(t-23, self.func(data, -23))
        self.assertEquals(t+1e20, self.func(data, 1e20))


class SumTortureTest(unittest.TestCase):
    def testTorture(self):
        # Tim Peters' torture test for sum, and variants of same.
        func = stats.sum
        self.assertEquals(func([1, 1e100, 1, -1e100]*10000), 20000.0)
        self.assertEquals(func([1e100, 1, 1, -1e100]*10000), 20000.0)
        self.assertAlmostEquals(
            func([1e-100, 1, 1e-100, -1]*10000), 2.0e-96, places=15)


class ProductTest(unittest.TestCase):

    def testEmpty(self):
        # Test that the empty product is 1.
        self.assertEquals(stats.product([]), 1)
        self.assertEquals(stats.product([], 123.456), 123.456)

    def testSorted(self):
        # Product shouldn't depend on the order of items.
        data = [i/7 for i in range(-3500, 3600, 100)]
        a = stats.product(data)
        random.shuffle(data)
        b = stats.product(data)
        self.assertEquals(a, b)

    def testZero(self):
        # Product of anything containing zero is always zero.
        for data in (range(23), range(-35, 36)):
            self.assertEquals(stats.product(data), 0)

    def testProduct(self):
        self.assertEquals(stats.product(range(1, 24)), math.factorial(23))

    def testExact(self):
        data = [i/(i+1) for i in range(1, 1024)]
        random.shuffle(data)
        self.assertAlmostEquals(stats.product(data), 1/1024, places=12)
        self.assertAlmostEquals(stats.product(data, 2.5), 5/2048, places=12)

    def testTypes(self):
        for data in (range(1, 42), range(-35, 36, 2)):
            a = stats.product(data)
            b = stats.product(list(data))
            c = stats.product(tuple(data))
            d = stats.product(iter(data))
            self.assertEquals(a, b)
            self.assertEquals(a, c)
            self.assertEquals(a, d)

    def testStart(self):
        data = [random.uniform(1, 50) for _ in range(10)]
        t = stats.product(data)
        # assertAlmostEquals not up to the job here.
        for start in (42, 0.2, -23, 1e20):
            a = t*start
            b = stats.product(data, start)
            err = abs(a - b)/b
            self.assert_(err <= 1e-12)


class SumSqTest(SumTest):
    def __init__(self, *args, **kwargs):
        SumTest.__init__(self, *args, **kwargs)
        self.func = stats.sumsq

    def testSum(self):
        data = [random.random() for _ in range(100)]
        self.assertEquals(self.func(data), math.fsum(x**2 for x in data))

    def testExact(self):
        # sum of squares of 1, 2, 3, ... n = n(n+1)(2n+1)/6
        data = range(1, 101)
        expected = 100*101*201/6
        self.assertEquals(self.func(data), expected)


class CumulativeSumTest(unittest.TestCase):
    def testGenerator(self):
        # Test that function is a generator.
        self.assert_(inspect.isgeneratorfunction(stats.cumulative_sum))

    def testFinal(self):
        # Test the final result has the expected value.
        data = [3.2*i - 12.3 for i in range(0, 35, 3)]
        random.shuffle(data)
        expected = stats.sum(data)
        results = list(stats.cumulative_sum(data))
        self.assertEquals(results[-1], expected)

    def testTorture(self):
        # Based on Tim Peters' torture test for sum.
        it = stats.cumulative_sum([1, 1e100, 1, -1e100]*10000)
        # Expect 1, 1e100, 1e100, 2, 3, 1e100, 1e100, 4, ... 20000
        shortsum = 0
        for i, x in enumerate(it):
            r = i%4
            if r in (0, 2):
                shortsum += 1
            if r in (1, 2):
                self.assertEquals(x, 1e100)
            else:
                self.assertEquals(x, shortsum)

    def testStart(self):
        data = list(range(1, 35))
        expected = 34*35/2  # Exact value.
        random.shuffle(data)
        results = list(stats.cumulative_sum(data))
        self.assertEquals(results[-1], expected)
        for start in (-2.5, 0.0, 1.0, 42, 56.789):
            results = list(stats.cumulative_sum(data, start))
            self.assertEquals(results[-1], expected+start)

    def testIteration(self):
        it = stats.cumulative_sum([])
        self.assertRaises(StopIteration, next, it)  #1
        it = stats.cumulative_sum([42])
        self.assertEquals(next(it), 42)
        self.assertRaises(StopIteration, next, it)  #2
        it = stats.cumulative_sum([42, 23])
        self.assertEquals(next(it), 42)
        self.assertEquals(next(it), 65)
        self.assertRaises(StopIteration, next, it)  #3

    def testIterationStart(self):
        it = stats.cumulative_sum([], 3)
        self.assertEquals(next(it), 3)
        self.assertRaises(StopIteration, next, it)  #1
        it = stats.cumulative_sum([42], 3)
        self.assertEquals(next(it), 45)
        self.assertRaises(StopIteration, next, it)  #2
        it = stats.cumulative_sum([42, 23], 3)
        self.assertEquals(next(it), 45)
        self.assertEquals(next(it), 68)
        self.assertRaises(StopIteration, next, it)  #3


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

    def testFailures(self):
        # Negative stdev or sample size is bad.
        self.assertRaises(ValueError, stats.sterrmean, -1, 2)
        self.assertRaises(ValueError, stats.sterrmean, -1, 2, 3)
        self.assertRaises(ValueError, stats.sterrmean, 1, -2, 3)
        self.assertRaises(ValueError, stats.sterrmean, 1, -2)

    def testPopulationSize(self):
        # Population size must not be less than sample size.
        self.assertRaises(ValueError, stats.sterrmean, 1, 100, 99)
        # But equal or greater is allowed.
        self.assert_(stats.sterrmean(1, 100, 100) or True)
        self.assert_(stats.sterrmean(1, 100, 101) or True)

    def testZeroStdev(self):
        for n in (5, 10, 25, 100):
            self.assertEquals(stats.sterrmean(0.0, n), 0.0)
            self.assertEquals(stats.sterrmean(0.0, n, n*10), 0.0)

    def testZeroSizes(self):
        for s in (0.1, 1.0, 32.1):
            x = stats.sterrmean(s, 0)
            self.assert_(math.isinf(x))
            x = stats.sterrmean(s, 0, 100)
            self.assert_(math.isinf(x))
            x = stats.sterrmean(s, 0, 0)
            self.assert_(math.isnan(x))

    def testResult(self):
        self.assertEquals(stats.sterrmean(0.25, 25), 0.05)
        self.assertEquals(stats.sterrmean(1.0, 100), 0.1)
        self.assertEquals(stats.sterrmean(2.5, 16), 0.625)

    def testFPC(self):
        self.assertAlmostEquals(
            stats.sterrmean(0.25, 25, 100), 0.043519413989, places=11)
        self.assertAlmostEquals(
            stats.sterrmean(1.0, 100, 150), 5.79284446364e-2, places=11)
        self.assertAlmostEquals(
            stats.sterrmean(2.5, 16, 20), 0.286769667338, places=11)


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
    # Check maximum error found in corr1.
    #
    err = stats._MAX_CORR1_ERR
    if err:
        print('Warning: corr1 residue found: %r' % err)
    #
    # Check for reference leaks.
    #
    gc.collect()
    if gc.garbage:
        print("List of uncollectable garbage:")
        print(gc.garbage)
    else:
        pr("No garbage found.")


