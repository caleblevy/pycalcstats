#!/usr/bin/env python3
"""Test suite for stats.py

Runs:
    doctests from the stats module
    tests from the examples text file (if any)
    unit tests in this module
    a limited test for uncollectable objects

"""

import doctest
import gc
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


# Module being tested.
import stats

print(stats.__file__)


# === Helper functions ===

# === Data sets for testing ===

# === Test suites ===

class GlobalTest(unittest.TestCase):
    """Test the state and/or existence of globals."""
    def testMeta(self):
        """Test existence of metadata."""
        attrs = ("__doc__ __version__ __date__ __author__"
                 " __author_email__ __all__").split()
        for meta in attrs:
            self.failUnless(hasattr(stats, meta), "missing %s" % meta)


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


class SortedDataDecoratorTest(unittest.TestCase):
    """Test that the sorted_data decorator works correctly."""
    def testDecorator(self):
        @stats.sorted_data
        def f(data):
            return data

        values = random.sample(range(1000), 100)
        result = f(values)
        self.assertEquals(result, sorted(values))


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


class VarianceTest(unittest.TestCase):
    data = (4.0, 7.0, 13.0, 16.0)
    expected = 30.0  # Expected (exact) sample variance.

    def test_small(self):
        self.assertEquals(stats.variance(self.data), self.expected)
    def test_big(self):
        data = [x + 1e6 for x in self.data]
        self.assertEquals(stats.variance(data), self.expected)
    def test_huge(self):
        data = [x + 1e9 for x in self.data]
        self.assertEquals(stats.variance(data), self.expected)


class CompareAgainstExternalResultsTest(unittest.TestCase):
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


class DrMathTests(unittest.TestCase):
    # Sample data for testing quartiles taken from Dr Math page:
    # http://mathforum.org/library/drmath/view/60969.html
    # FIXME results given doesn't include Q2 points.
    A = range(1, 9)
    B = range(1, 10)
    C = range(1, 11)
    D = range(1, 12)

    def testTukey(self):
        f = stats._quartiles_tukey
        q1, q2, q3 = f(self.A)
        self.assertEquals(q1, 2.5)
        self.assertEquals(q3, 6.5)
        q1, q2, q3 = f(self.B)
        self.assertEquals(q1, 3.0)
        self.assertEquals(q3, 7.0)
        q1, q2, q3 = f(self.C)
        self.assertEquals(q1, 3.0)
        self.assertEquals(q3, 8.0)
        q1, q2, q3 = f(self.D)
        self.assertEquals(q1, 3.5)
        self.assertEquals(q3, 8.5)

    def testMM(self):
        f = stats._quartiles_mm
        q1, q2, q3 = f(self.A)
        self.assertEquals(q1, 2.5)
        self.assertEquals(q3, 6.5)
        q1, q2, q3 = f(self.B)
        self.assertEquals(q1, 2.5)
        self.assertEquals(q3, 7.5)
        q1, q2, q3 = f(self.C)
        self.assertEquals(q1, 3.0)
        self.assertEquals(q3, 8.0)
        q1, q2, q3 = f(self.D)
        self.assertEquals(q1, 3.0)
        self.assertEquals(q3, 9.0)

    def testMS(self):
        f = stats._quartiles_ms
        q1, q2, q3 = f(self.A)
        self.assertEquals(q1, 2)
        self.assertEquals(q3, 7)
        q1, q2, q3 = f(self.B)
        self.assertEquals(q1, 3)
        self.assertEquals(q3, 7)
        q1, q2, q3 = f(self.C)
        self.assertEquals(q1, 3)
        self.assertEquals(q3, 8)
        q1, q2, q3 = f(self.D)
        self.assertEquals(q1, 3)
        self.assertEquals(q3, 9)

    def testMinitab(self):
        f = stats._quartiles_minitab
        q1, q2, q3 = f(self.A)
        self.assertEquals(q1, 2.25)
        self.assertEquals(q3, 6.75)
        q1, q2, q3 = f(self.B)
        self.assertEquals(q1, 2.5)
        self.assertEquals(q3, 7.5)
        q1, q2, q3 = f(self.C)
        self.assertEquals(q1, 2.75)
        self.assertEquals(q3, 8.25)
        q1, q2, q3 = f(self.D)
        self.assertEquals(q1, 3.0)
        self.assertEquals(q3, 9.0)

    def testExcel(self):
        f = stats._quartiles_excel
        q1, q2, q3 = f(self.A)
        self.assertEquals(q1, 2.75)
        self.assertEquals(q3, 6.25)
        q1, q2, q3 = f(self.B)
        self.assertEquals(q1, 3.0)
        self.assertEquals(q3, 7.0)
        q1, q2, q3 = f(self.C)
        self.assertEquals(q1, 3.25)
        self.assertEquals(q3, 7.75)
        q1, q2, q3 = f(self.D)
        self.assertEquals(q1, 3.5)
        self.assertEquals(q3, 8.5)


class QuartileTest(unittest.TestCase):
    func = stats.quartiles

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        # Black magic to force self.func to be a function rather
        # than a method.
        self.func = self.__class__.func

    def testSorting(self):
        """Test that quartiles doesn't sort in place."""
        data = [2, 4, 1, 3, 0, 5]
        assert data != sorted(data)
        save = data[:]
        assert save is not data
        _ = self.func(data)
        self.assertEquals(data, save)

    def testTooFewItems(self):
        self.assertRaises(ValueError, self.func, [])
        self.assertRaises(ValueError, self.func, [1])
        self.assertRaises(ValueError, self.func, [1, 2])

    def testUnsorted(self):
        data = [3, 4, 2, 1, 0, 5]
        assert data != sorted(data)
        self.assertEquals(self.func(data), (1, 2.5, 4))

    def testIter(self):
        self.assertEquals(self.func(range(12)), (2.5, 5.5, 8.5))
        self.assertEquals(self.func(range(13)), (2.5, 6, 9.5))
        self.assertEquals(self.func(range(14)), (3, 6.5, 10))
        self.assertEquals(self.func(range(15)), (3, 7, 11))

    def testSmall(self):
        data = [0, 1, 2]
        self.assertEquals(self.func(data), (0, 1, 2))
        data.append(3)
        self.assertEquals(self.func(data), (0.5, 1.5, 2.5))
        data.append(4)
        self.assertEquals(self.func(data), (0.5, 2, 3.5))
        data.append(5)
        self.assertEquals(self.func(data), (1, 2.5, 4))
        data.append(6)
        self.assertEquals(self.func(data), (1, 3, 5))

    def testBig(self):
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


