#!/usr/bin/env python3

##  Copyright (c) 2011 Steven D'Aprano.
##  See the file stats/__init__.py for the licence terms for this software.

"""
Test suite for the stats.univar module.

"""

import math
import random
import unittest

from stats._tests import NumericTestCase
import stats._tests.common as common
import stats._tests.basic

# The module to be tested:
import stats.univar


@unittest.skip('geometric mean currently too inaccurate')
class GeometricMeanTest(stats._tests.basic.MeanTest):
    rel = 1e-11

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.univar.geometric_mean
        self.expected = 4.56188290183

    def testNegative(self):
        data = [1.0, 2.0, -3.0, 4.0]
        assert any(x < 0.0 for x in data)
        self.assertRaises(ValueError, self.func, data)

    def testZero(self):
        data = [1.0, 2.0, 0.0, 4.0]
        assert any(x == 0.0 for x in data)
        self.assertEqual(self.func(data), 0.0)


class HarmonicMeanTest(stats._tests.basic.MeanTest):
    rel = 1e-8

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.univar.harmonic_mean
        self.expected = 3.4995090404755

    def testNegative(self):
        # The harmonic mean of negative numbers is allowed.
        data = [1.0, -2.0, 4.0, -8.0]
        assert any(x < 0.0 for x in data)
        self.assertEqual(self.func(data), 4*8/5)

    def testZero(self):
        # The harmonic mean of anything with a zero in it should be zero.
        data = [1.0, 2.0, 0.0, 4.0]
        assert any(x == 0.0 for x in data)
        self.assertEqual(self.func(data), 0.0)
        # FIX ME test for signed zeroes?


class QuadraticMeanTest(stats._tests.basic.MeanTest):
    rel = 1e-8

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.univar.quadratic_mean
        self.expected = 6.19004577259

    def testNegative(self):
        data = [-x for x in self.data]
        self.assertApproxEqual(self.func(data), self.expected)
        data = [1.0, -2.0, -3.0, 4.0]
        self.assertEqual(self.func(data), math.sqrt(30/4))

    def testZero(self):
        data = [1.0, 2.0, 0.0, 4.0]
        assert any(x == 0.0 for x in data)
        self.assertEqual(self.func(data), math.sqrt(21/4))


class ModeTest(NumericTestCase):
    # FIX ME incomplete test coverage

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.univar.mode
        self.data = [
                    1.1,
                    2.2, 2.2,
                    3.3,
                    4.4, 4.4, 4.4,
                    5.5, 5.5, 5.5, 5.5,
                    6.6, 6.6,
                    7.7,
                    8.8,
                    ]
        self.expected = 5.5

    def testModeless(self):
        data = list(set(self.data))
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


class AverageDeviationTest(NumericTestCase, common.UnivariateMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.univar.average_deviation
        self.extras = [(), (1,), (-2.5,), (None,)]

    def testSuppliedMean(self):
        # Test that pre-calculating the mean gives the same result.
        for data in (range(35), range(-17, 53, 7), range(11, 79, 3)):
            data = list(data)
            random.shuffle(data)
            m = stats.mean(data)
            result1 = self.func(data)
            result2 = self.func(data, m)
            self.assertEqual(result1, result2)

    def testSingleton(self):
        self.assertEqual(self.func([42]), 0)
        self.assertEqual(self.func([42], 40), 2)

    def testMain(self):
        data = [-1.25, 0.5, 0.5, 1.75, 3.25, 4.5, 4.5, 6.25, 6.75, 9.75]
        expected = 2.7
        for delta in (0, 100, 1e6, 1e9):
            self.assertEqual(self.func(x+delta for x in data), expected)


class MedianAverageDeviationTest(NumericTestCase, common.UnivariateMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.univar.median_average_deviation
        self.extras = [
            (), (1,), (-2.5,), (None,), (4.5, 1), (7.5, -1),
            (4.5, 1, 1), (4.5, -1, 2), (4.5, 0, -2),
            ]

    def testSuppliedMedian(self):
        # Test that pre-calculating the median gives the same result.
        import stats.order
        for data in (range(35), range(-17, 53, 7), range(11, 79, 3)):
            result1 = self.func(data)
            m = stats.order.median(data)
            data = list(data)
            random.shuffle(data)
            result2 = self.func(data, m)
            self.assertEqual(result1, result2)

    def testMain(self):
        data = [-1.25, 0.5, 0.5, 1.75, 3.25, 4.5, 4.5, 6.25, 6.75, 9.75]
        expected = 2.625
        for delta in (0, 100, 1e6, 1e9):
            self.assertEqual(self.func(x+delta for x in data), expected)

    def testHasScaling(self):
        self.assertTrue(hasattr(self.func, 'scaling'))

    def testNoScaling(self):
        # Test alternative ways of spelling no scaling factor.
        data = [random.random()+23 for _ in range(100)]
        expected = self.func(data)
        for scale in (1, None, 'none'):
            self.assertEqual(self.func(data, scale=scale), expected)

    def testScales(self):
        data = [100*random.random()+42 for _ in range(100)]
        expected = self.func(data)
        self.assertEqual(self.func(data, scale='normal'), expected*1.4826)
        self.assertApproxEqual(
            self.func(data, scale='uniform'),
            expected*1.1547, # Documented value in docstring.
            tol=0.0001, rel=None)
        self.assertEqual(self.func(data, scale='uniform'),
            expected*math.sqrt(4/3))  # Exact value.
        for x in (-1.25, 0.0, 1.25, 4.5, 9.75):
            self.assertEqual(self.func(data, scale=x), expected*x)

    def testCaseInsensitiveScaling(self):
        for scale in ('normal', 'uniform', 'none'):
            data = [67*random.random()+19 for _ in range(100)]
            a = self.func(data, scale=scale.lower())
            b = self.func(data, scale=scale.upper())
            c = self.func(data, scale=scale.title())
            self.assertEqual(a, b)
            self.assertEqual(a, c)

    def testSignOdd(self):
        data = [23*random.random()+42 for _ in range(55)]
        assert len(data)%2 == 1
        a = self.func(data, sign=-1)
        b = self.func(data, sign=0)
        c = self.func(data, sign=1)
        d = self.func(data)
        self.assertEqual(a, b)
        self.assertEqual(a, c)
        self.assertEqual(a, d)

    def testSignEven(self):
        data = [0.5, 1.5, 3.25, 4.25, 6.25, 6.75]
        assert len(data)%2 == 0
        self.assertEqual(self.func(data), 2.375)
        self.assertEqual(self.func(data, sign=-1), 1.75)
        self.assertEqual(self.func(data, sign=0), 2.375)
        self.assertEqual(self.func(data, sign=1), 2.5)


class PearsonModeSkewnessTest(NumericTestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.univar.pearson_mode_skewness

    def testFailure(self):
        # Test that stdev must be positive.
        self.assertRaises(ValueError, self.func, 2, 3, -1)
        self.assertRaises(ValueError, self.func, 3, 2, -5)

    def testNan(self):
        # Test that a numerator and denominator of zero returns NAN.
        self.assertTrue(math.isnan(self.func(5, 5, 0)))
        self.assertTrue(math.isnan(self.func(42, 42, 0)))

    def testInf(self):
        # Test that a non-zero numerator and zero denominator returns INF.
        self.assertTrue(math.isinf(self.func(3, 2, 0)))
        self.assertTrue(math.isinf(self.func(2, 3, 0)))

    def testZero(self):
        # Test that a zero numerator and non-zero denominator returns zero.
        self.assertEqual(self.func(3, 3, 1), 0)
        self.assertEqual(self.func(42, 42, 7), 0)

    def testSkew(self):
        # Test skew calculations.
        self.assertEqual(self.func(2.5, 2.25, 2.5), 0.1)
        self.assertEqual(self.func(225, 250, 25), -1.0)


class SkewnessTest(NumericTestCase):
    # FIXME incomplete test cases

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.univar.skewness
        self.extras = [(), (None, None), (1.0, 0.1)]

    def test_uniform(self):
        # Compare the calculated skewness against an exact result
        # calculated from a uniform distribution.
        data = range(10000)
        self.assertEqual(self.func(data), 0.0)
        data = [x + 1e9 for x in data]
        self.assertEqual(self.func(data), 0.0)

    def test_shift1(self):
        data = [(2*i+1)/4 for i in range(1000)]
        random.shuffle(data)
        k1 = self.func(data)
        self.assertEqual(k1, 0.0)
        k2 = self.func(x+1e9 for x in data)
        self.assertEqual(k2, 0.0)

    def test_shift2(self):
        d1 = [(2*i+1)/3 for i in range(1000)]
        d2 = [(3*i-19)/2 for i in range(1000)]
        data = [x*y for x,y in zip(d1, d2)]
        random.shuffle(data)
        k1 = self.func(data)
        k2 = self.func(x+1e9 for x in data)
        self.assertApproxEqual(k1, k2, tol=1e-7)

    def testMeanStdev(self):
        # Giving the sample mean and/or stdev shouldn't change the result.
        d1 = [(98-3*i)/6 for i in range(100)]
        d2 = [(14*i-3)/2 for i in range(100)]
        random.shuffle(d1)
        random.shuffle(d2)
        data = [x*y for x,y in zip(d1, d2)]
        m = stats.mean(data)
        s = stats.stdev(data)
        a = self.func(data)
        b = self.func(data, m)
        c = self.func(data, None, s)
        d = self.func(data, m, s)
        self.assertEqual(a, b)
        self.assertEqual(a, c)
        self.assertEqual(a, d)


class KurtosisTest(NumericTestCase):
    # FIXME incomplete test cases
    tol = 1e-7
    rel = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.univar.kurtosis
        self.extras = [(), (None, None), (1.0, 0.1)]

    def corrected_uniform_kurtosis(self, n):
        """Return the exact kurtosis for a discrete uniform distribution."""
        # Calculate the exact population kurtosis:
        expected = -6*(n**2 + 1)/(5*(n - 1)*(n + 1))
        # Give a correction factor to adjust it for sample kurtosis:
        expected *= (n/(n-1))**3
        return expected

    def test_uniform(self):
        # Compare the calculated kurtosis against an exact result
        # calculated from a uniform distribution.
        n = 10000
        data = range(n)
        expected = self.corrected_uniform_kurtosis(n)
        self.assertApproxEqual(self.func(data), expected)
        data = [x + 1e9 for x in data]
        self.assertApproxEqual(self.func(data), expected)

    def test_shift1(self):
        data = [(2*i+1)/4 for i in range(1000)]
        random.shuffle(data)
        k1 = self.func(data)
        k2 = self.func(x+1e9 for x in data)
        self.assertEqual(k1, k2)

    def test_shift2(self):
        d1 = [(2*i+1)/3 for i in range(1000)]
        d2 = [(3*i-19)/2 for i in range(1000)]
        random.shuffle(d1)
        random.shuffle(d2)
        data = [x*y for x,y in zip(d1, d2)]
        k1 = self.func(data)
        k2 = self.func(x+1e9 for x in data)
        self.assertApproxEqual(k1, k2, tol=1e-9)

    def testMeanStdev(self):
        # Giving the sample mean and/or stdev shouldn't change the result.
        d1 = [(17*i-45)/16 for i in range(100)]
        d2 = [(9*i-25)/3 for i in range(100)]
        random.shuffle(d1)
        random.shuffle(d2)
        data = [x*y for x,y in zip(d1, d2)]
        m = stats.mean(data)
        s = stats.stdev(data)
        a = self.func(data)
        b = self.func(data, m)
        c = self.func(data, None, s)
        d = self.func(data, m, s)
        self.assertEqual(a, b)
        self.assertEqual(a, c)
        self.assertEqual(a, d)


class ProductTest(NumericTestCase):
    # FIXME incomplete test cases

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.univar.product
        self.extras = [(), (0.1,), (2.5,), (-3.5,)]

    def testEmpty(self):
        # Test that the empty product is 1, or whatever value is specified.
        self.assertEqual(self.func([]), 1)
        self.assertEqual(self.func([], 123.456), 123.456)

    def testZero(self):
        # Product of anything containing zero is always zero.
        for data in (range(23), range(-35, 36)):
            self.assertEqual(self.func(data), 0)

    def testNegatives(self):
        self.assertEqual(self.func([2, -3]), -6)
        self.assertEqual(self.func([2, -3, -4]), 24)
        self.assertEqual(self.func([-2, 3, -4, -5]), -120)

    def testProduct(self):
        self.assertEqual(self.func([1.5, 5.0, 7.5, 12.0]), 675.0)
        self.assertEqual(self.func([5]*20), 5**20)
        self.assertEqual(self.func(range(1, 24)), math.factorial(23))
        data = [i/(i+1) for i in range(1, 1024)]
        random.shuffle(data)
        self.assertApproxEqual(self.func(data), 1/1024, tol=1e-12)
        self.assertApproxEqual(self.func(data, 2.5), 5/2048, tol=1e-12)

    def testStart(self):
        data = [random.uniform(1, 50) for _ in range(10)]
        t = self.func(data)
        for start in (42, 0.2, -23, 1e20):
            a = t*start
            b = self.func(data, start)
            self.assertApproxEqual(a, b, rel=1e-12)

    def testTorture(self):
        # Torture test for product.
        data = []
        for i in range(1, 101):
            data.append(i)
            data.append(1/i)
        self.assertApproxEqual(self.func(data), 1.0, tol=1e-14)
        random.shuffle(data)
        self.assertApproxEqual(self.func(data), 1.0, tol=1e-14)


class StErrMeanTest(NumericTestCase):
    tol=1e-11

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.univar.sterrmean

    def testBadStdev(self):
        # Negative stdev is bad.
        self.assertRaises(ValueError, self.func, -1, 2)
        self.assertRaises(ValueError, self.func, -1, 2, 3)

    def testBadSizes(self):
        # Negative sample or population sizes are bad.
        self.assertRaises(ValueError, self.func, 1, -2)
        self.assertRaises(ValueError, self.func, 1, -2, 3)
        self.assertRaises(ValueError, self.func, 1, 2, -3)
        # So are fractional sizes.
        self.assertRaises(ValueError, self.func, 1, 2.5)
        self.assertRaises(ValueError, self.func, 1, 2.5, 3)
        self.assertRaises(ValueError, self.func, 1, 2.5, 3.5)
        self.assertRaises(ValueError, self.func, 1, 2, 3.5)

    def testPopulationSize(self):
        # Population size must not be less than sample size.
        self.assertRaises(ValueError, self.func, 1, 100, 99)
        # But equal or greater is allowed.
        self.assertEqual(self.func(1, 100, 100), 0.0)
        self.assertTrue(self.func(1, 100, 101))

    def testZeroStdev(self):
        for n in (5, 10, 25, 100):
            self.assertEqual(self.func(0.0, n), 0.0)
            self.assertEqual(self.func(0.0, n, n*10), 0.0)

    def testZeroSizes(self):
        for s in (0.1, 1.0, 32.1):
            x = self.func(s, 0)
            self.assertTrue(math.isinf(x))
            x = self.func(s, 0, 100)
            self.assertTrue(math.isinf(x))
            x = self.func(s, 0, 0)
            self.assertTrue(math.isnan(x))

    def testResult(self):
        self.assertEqual(self.func(0.25, 25), 0.05)
        self.assertEqual(self.func(1.0, 100), 0.1)
        self.assertEqual(self.func(2.5, 16), 0.625)

    def testFPC(self):
        self.assertApproxEqual(
            self.func(0.25, 25, 100), 0.043519413989)
        self.assertApproxEqual(
            self.func(1.0, 100, 150), 5.79284446364e-2)
        self.assertApproxEqual(
            self.func(2.5, 16, 20), 0.286769667338)


class StErrSkewnessTest(NumericTestCase):
    tol=1e-12

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.univar.sterrskewness

    def testBadSize(self):
        # Negative sample size is bad.
        self.assertRaises(ValueError, self.func, -2)
        # So is fractional sample size.
        self.assertRaises(ValueError, self.func, 2.5)

    def testZero(self):
        x = self.func(0)
        self.assertTrue(math.isinf(x))

    def testResult(self):
        self.assertEqual(self.func(6), 1.0)
        self.assertApproxEqual(self.func(10), 0.774596669241)
        self.assertApproxEqual(self.func(20), 0.547722557505)
        self.assertEqual(self.func(24), 0.5)
        self.assertApproxEqual(self.func(55), 0.330289129538)


class StErrKurtosisTest(NumericTestCase):
    tol=1e-12

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.univar.sterrkurtosis

    def testBadSize(self):
        # Negative sample size is bad.
        self.assertRaises(ValueError, self.func, -2)
        # So is fractional sample size.
        self.assertRaises(ValueError, self.func, 2.5)

    def testZero(self):
        x = self.func(0)
        self.assertTrue(math.isinf(x))

    def testResult(self):
        self.assertEqual(self.func(6), 2.0)
        self.assertApproxEqual(self.func(10), 1.54919333848)
        self.assertApproxEqual(self.func(20), 1.09544511501)
        self.assertEqual(self.func(24), 1.0)
        self.assertApproxEqual(self.func(55), 0.660578259076)


class CircularMeanTest(NumericTestCase, common.UnivariateMixin):
    tol = 1e-12

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.univar.circular_mean

    def testDefaultDegrees(self):
        # Test that degrees are the default.
        data = [355, 5, 15, 320, 45]
        theta = self.func(data)
        phi = self.func(data, True)
        assert self.func(data, False) != theta
        self.assertEqual(theta, phi)

    def testRadians(self):
        # Test that degrees and radians (usually) give different results.
        data = [355, 5, 15, 320, 45]
        a = self.func(data, True)
        b = self.func(data, False)
        self.assertNotEquals(a, b)

    def testSingleton(self):
        for x in (-1.0, 0.0, 1.0, 3.0):
            self.assertEqual(self.func([x], False), x)
            self.assertApproxEqual(self.func([x], True), x)

    def testNegatives(self):
        data1 = [355, 5, 15, 320, 45]
        theta = self.func(data1)
        data2 = [d-360 if d > 180 else d for d in data1]
        phi = self.func(data2)
        self.assertApproxEqual(theta, phi)

    def testIter(self):
        theta = self.func(iter([355, 5, 15]))
        self.assertApproxEqual(theta, 5.0)

    def testSmall(self):
        t = self.func([0, 360])
        self.assertApproxEqual(t, 0.0)
        t = self.func([10, 20, 30])
        self.assertApproxEqual(t, 20.0)
        t = self.func([355, 5, 15])
        self.assertApproxEqual(t, 5.0)

    def testFullCircle(self):
        # Test with angle > full circle.
        theta = self.func([3, 363])
        self.assertApproxEqual(theta, 3)

    def testBig(self):
        pi = math.pi
        # Generate angles between pi/2 and 3*pi/2, with expected mean of pi.
        delta = pi/1000
        data = [pi/2 + i*delta for i in range(1000)]
        data.append(3*pi/2)
        assert data[0] == pi/2
        assert len(data) == 1001
        random.shuffle(data)
        theta = self.func(data, False)
        self.assertApproxEqual(theta, pi)
        # Now try the same with angles in the first and fourth quadrants.
        data = [0.0]
        for i in range(1, 501):
            data.append(i*delta)
            data.append(2*pi - i*delta)
        assert len(data) == 1001
        random.shuffle(data)
        theta = self.func(data, False)
        self.assertApproxEqual(theta, 0.0)

