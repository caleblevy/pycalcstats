#!/usr/bin/env python3

##  Copyright (c) 2011 Steven D'Aprano.
##  See the file stats/__init__.py for the licence terms for this software.

"""
Test suite for the stats module (stats.__init__.py).

"""

import math
import random
import unittest

from stats.tests import NumericTestCase
import stats.tests.common as common

# The module to be tested:
import stats


class GlobalsTest(unittest.TestCase, common.GlobalsMixin):
    module = stats

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expected_metadata = common.GlobalsMixin.expected_metadata[:]
        self.expected_metadata.extend(
            "__version__ __date__ __author__ __author_email__".split()
            )


class SumTest(
    NumericTestCase, common.SingleDataPassMixin, common.UnivariateMixin
    ):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.sum

    def testEmptyData(self):
        # Override method from UnivariateMixin.
        for empty in ([], (), iter([])):
            self.assertEqual(0, self.func(empty))

    def testEmptySum(self):
        # Test the value of the empty sum.
        self.assertEqual(self.func([]), 0)
        self.assertEqual(self.func([], 123.456), 123.456)

    def testSum(self):
        # Compare with the math.fsum function.
        data = [random.uniform(-100, 1000) for _ in range(1000)]
        self.assertEqual(self.func(data), math.fsum(data))

    def testExact(self):
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

    def testStart(self):
        data = [random.uniform(1, 1000) for _ in range(100)]
        t = self.func(data)
        self.assertEqual(t+42, self.func(data, 42))
        self.assertEqual(t-23, self.func(data, -23))
        self.assertEqual(t+1e20, self.func(data, 1e20))


class SumTortureTest(NumericTestCase):
    def testTorture(self):
        # Tim Peters' torture test for sum, and variants of same.
        func = stats.sum
        self.assertEqual(func([1, 1e100, 1, -1e100]*10000), 20000.0)
        self.assertEqual(func([1e100, 1, 1, -1e100]*10000), 20000.0)
        self.assertApproxEqual(
            func([1e-100, 1, 1e-100, -1]*10000), 2.0e-96, tol=1e-15)


class MeanTest(
    NumericTestCase, common.SingleDataPassMixin, common.UnivariateMixin
    ):
    tol = rel = None  # Default to expect exact equality.

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
        assert expected != 1e9  # Avoid catastrophic loss of precision.
        self.assertApproxEqual(self.func(data), expected)

    def testIter(self):
        self.assertApproxEqual(self.func(iter(self.data)), self.expected)

    def testSingleton(self):
        for x in self.data:
            self.assertApproxEqual(self.func([x]), x)

    def testDoubling(self):
        # Average of [a,b,c...z] should be same as for [a,a,b,b,c,c...z,z].
        data = [random.random() for _ in range(1000)]
        a = self.func(data)
        b = self.func(data*2)
        self.assertEqual(a, b)


"""
class TinyVariance(NumericTestCase):
    # Minimal tests for variance and friends.
   def testVariance(self):
       data = [1, 2, 3]
       assert stats.mean(data) == 2
       self.assertEqual(stats.pvariance(data), 2/3)
       self.assertEqual(stats.variance(data), 1.0)
       self.assertEqual(stats.pvariance1(data), 2/3)
       self.assertEqual(stats.variance1(data), 1.0)
       self.assertEqual(stats.pstdev(data), math.sqrt(2/3))
       self.assertEqual(stats.stdev(data), 1.0)
       self.assertEqual(stats.pstdev1(data), math.sqrt(2/3))
       self.assertEqual(stats.stdev1(data), 1.0)


class PVarianceTest(NumericTestCase):
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

    tol = 1e-16  # Absolute error accepted.

    def __init__(self, *args, **kwargs):
        NumericTestCase.__init__(self, *args, **kwargs)
        # Force self.func to be a function rather than a method.
        self.func = self.__class__.func

    def testEmptyFailure(self):
        for data in ([], (), iter([])):
            self.assertRaises(ValueError, self.func, data)

    def test_small(self):
        self.assertEqual(self.func(self.data), self.expected)

    def test_big(self):
        data = [x + 1e6 for x in self.data]
        self.assertEqual(self.func(data), self.expected)

    def test_huge(self):
        data = [x + 1e9 for x in self.data]
        self.assertEqual(self.func(data), self.expected)

    def test_uniform(self):
        # Compare the calculated variance against an exact result.
        self.assertEqual(self.func(self.uniform_data), self.uniform_expected)

    def testCompareHP(self):
        # Compare against a result calculated with a HP-48GX calculator.
        data = (list(range(1, 11)) + list(range(1000, 1201)) +
            [0, 3, 7, 23, 42, 101, 111, 500, 567])
        random.shuffle(data)
        self.assertApproxEqual(self.func(data), self.hp_expected)

    def testDuplicate(self):
        data = [random.uniform(-100, 500) for _ in range(20)]
        a = self.func(data)
        b = self.func(data*2)
        self.assertApproxEqual(a*self.scale, b)

    def testDomainError(self):
        # Domain error exception reported by Geremy Condra.
        data = [0.123456789012345]*10000
        # All the items are identical, so variance should be zero.
        self.assertApproxEqual(self.func(data), 0.0)

    def testWithLargeData(self):
        small_data = [random.gauss(7.5, 5.5) for _ in range(1000)]
        a = self.func(small_data)
        # We expect a to be close to the exact result for the variance,
        # namely 5.5**2, but if it's not, that's just a fluke of the random
        # sample. Either way, it doesn't matter.
        b = self.func(small_data)
        self.assertApproxEqual(a, b, tol=1e-12)

        def big_data():
            for _ in range(100):
                for x in small_data:
                    yield x

        c = self.func(big_data())
        # In principle, the calculated variance should be unchanged;
        # however due to rounding errors it may have changed somewhat.
        self.assertApproxEqual(a, c, tol=None, rel=0.001)


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
        expected = 57563.550
        self.assertApproxEqual(self.func(data), expected, tol=1e-3)
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
        self.assertApproxEqual(self.func(data), expected, tol=1e-4)


class VarianceMeanTest(NumericTestCase):
    # Test variance calculations when the mean is explicitly supplied.

    def compare_with_and_without_mean(self, func):
        mu = 100*random.random()
        sigma = 10*random.random()+1
        for data in (
            [-6, -2, 0, 3, 4, 5, 5, 5, 6, 7, 9, 11, 15, 25, 26, 27, 28, 42],
            [random.random() for _ in range(10)],
            [random.uniform(10000, 11000) for _ in range(50)],
            [random.gauss(mu, sigma) for _ in range(50)],
            ):
            m = stats.mean(data)
            a = func(data)
            b = func(data, m)
            self.assertApproxEqual(a, b, tol=None, rel=None)

    def test_pvar(self):
        self.compare_with_and_without_mean(stats.pvariance)

    def test_var(self):
        self.compare_with_and_without_mean(stats.variance)

    def test_pstdev(self):
        self.compare_with_and_without_mean(stats.pstdev)

    def test_stdev(self):
        self.compare_with_and_without_mean(stats.stdev)


"""

