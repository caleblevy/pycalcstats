#!/usr/bin/env python3

##  Copyright (c) 2011 Steven D'Aprano.
##  See the file stats/__init__.py for the licence terms for this software.

"""
Test suite for the stats module (stats.__init__.py).

"""

import math
import random
import unittest

from stats._tests import NumericTestCase
import stats._tests.common as common

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


class SumTest(NumericTestCase, common.UnivariateMixin):
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


class MeanTest(NumericTestCase, common.UnivariateMixin):
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


class MinimalVarianceTest(NumericTestCase):
    # Minimal tests for variance and friends.

   def testVariance(self):
       data = [1, 2, 3]
       assert stats.mean(data) == 2
       self.assertEqual(stats.pvariance(data), 2/3)
       self.assertEqual(stats.variance(data), 1.0)
       self.assertEqual(stats.pstdev(data), math.sqrt(2/3))
       self.assertEqual(stats.stdev(data), 1.0)


class PVarianceTest(NumericTestCase, common.UnivariateMixin):
    # Test population variance.

    tol = 1e-16  # Absolute error accepted.

    def __init__(self, *args, **kwargs):
        NumericTestCase.__init__(self, *args, **kwargs)
        self.func = stats.pvariance
        # Standard test data.
        self.data = [4.0, 7.0, 13.0, 16.0]
        self.expected = 22.5  # Exact population variance of self.data.
        # Test data for exact (uniform distribution) test:
        self.uniform_data = range(10000)
        self.uniform_expected = (10000**2 - 1)/12
        # Expected result calculated by HP-48GX:
        self.hp_expected = 88349.2408884
        # Scaling factor when you duplicate each data point:
        self.scale = 1.0

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
        expected = self.func(data)*self.scale
        actual = self.func(data*2)
        self.assertApproxEqual(actual, expected)

    def testDomainError(self):
        # Domain error exception reported by Geremy Condra.
        data = [0.123456789012345]*10000
        # All the items are identical, so variance should be zero.
        self.assertApproxEqual(self.func(data), 0.0)


class PVarianceDupsTest(NumericTestCase):
    def testManyDuplicates(self):
        from stats import pvariance
        # Start with 1000 normally distributed data points.
        data = [random.gauss(7.5, 5.5) for _ in range(1000)]
        expected = pvariance(data)
        # We expect a to be close to the exact result for the variance,
        # namely 5.5**2, but because it's random, it might not be.
        # Either way, it doesn't matter.

        # Duplicating the data points should keep the variance the same.
        for n in (3, 5, 10, 20, 30):
            d = data*n
            actual = pvariance(d)
            self.assertApproxEqual(actual, expected, tol=1e-12)

        # Now try again with a lot of duplicates.
        def big_data():
            for _ in range(500):
                for x in data:
                    yield x

        actual = pvariance(big_data())
        self.assertApproxEqual(actual, expected, tol=1e-12)


class VarianceTest(common.SingleDataFailMixin, PVarianceTest):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.variance
        self.expected = 30.0  # Exact sample variance of self.data.
        self.uniform_expected = self.uniform_expected * 10000/(10000-1)
        self.hp_expected = 88752.6620797
        # Scaling factor when you duplicate each data point:
        self.scale = (2*20-2)/(2*20-1)

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.pstdev
        self.expected = math.sqrt(self.expected)
        self.uniform_expected = math.sqrt(self.uniform_expected)
        self.hp_expected = 297.236002006
        self.scale = math.sqrt(self.scale)


class StdevTest(VarianceTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.stdev
        self.expected = math.sqrt(self.expected)
        self.uniform_expected = math.sqrt(self.uniform_expected)
        self.hp_expected = 297.913850097
        self.scale = math.sqrt(self.scale)

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

