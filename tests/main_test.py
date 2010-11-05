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



class QuartileTest(unittest.TestCase):
    def testSorting(self):
        """Test that quartiles doesn't sort in place."""
        data = [2, 4, 1, 3, 0, 5]
        assert data != sorted(data)
        save = data[:]
        assert save is not data
        _ = stats.quartiles(data)
        self.assertEquals(data, save)

    def testTooFewItems(self):
        self.assertRaises(ValueError, stats.quartiles, [])
        self.assertRaises(ValueError, stats.quartiles, [1])
        self.assertRaises(ValueError, stats.quartiles, [1, 2])

    def testUnsorted(self):
        data = [3, 2, 1, 0, 5]
        assert data != sorted(data)
        self.assertEquals(stats.quartiles(data), (1, 2.5, 4))

    def testSmall(self):
        data = [0, 1, 2]
        self.assertEquals(stats.quartiles(data), (0, 1, 2))
        data.append(3)
        self.assertEquals(stats.quartiles(data), (0.5, 1.5, 2.5))
        data.append(4)
        self.assertEquals(stats.quartiles(data), (0.5, 2, 3.5))
        data.append(5)
        self.assertEquals(stats.quartiles(data), (1, 2.5, 4))
        data.append(6)
        self.assertEquals(stats.quartiles(data), (1, 3, 5))

    def testBig(self):
        data = list(range(1000, 2000))
        assert len(data) == 1000
        assert len(data)%4 == 0
        random.shuffle(data)
        self.assertEquals(stats.quartiles(data), (1249.5, 1499.5, 1749.5))
        data.append(2000)
        random.shuffle(data)
        self.assertEquals(stats.quartiles(data), (1249.5, 1500, 1749.5))
        data.append(2001)
        random.shuffle(data)
        self.assertEquals(stats.quartiles(data), (1249.5, 1500.5, 1749.5))



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


