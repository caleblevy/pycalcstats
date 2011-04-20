#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##  Copyright (c) 2011 Steven D'Aprano.
##  See the file stats/__init__.py for the licence terms for this software.

"""Test suite for the rest of the stats package."""

# Implementation note: many test results have been calculated using a
# HP-48GX calculator. Any reference to "RPL" refers to programs written
# on the HP-48GX.


import inspect
import math
import random
import unittest

import stats
import test_stats

# Modules to test:
import stats.co
import stats.multivar
import stats.order
import stats.univar



# === Mixin classes ===

class TestConsumerMixin:
    def testIsConsumer(self):
        # Test that the function is a consumer.
        cr = self.func()
        self.assertTrue(hasattr(cr, 'send'))



# === Unit tests ===

# -- co module --------------------------------------------------------

class CoGlobalsTest(test_stats.GlobalsTest):
    module = stats.co


class CoFeedTest(unittest.TestCase, TestConsumerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Define a coroutine.
        def counter():
            # Coroutine that counts items sent in.
            c = 0
            _ = (yield None)
            while True:
                c += 1
                _ = (yield c)

        self.func = counter

    def testIsGenerator(self):
        # A bare coroutine without the @coroutine decorator will be seen
        # as a generator, due to the presence of `yield`.
        self.assertTrue(inspect.isgeneratorfunction(self.func))

    def testCoroutine(self):
        # Test the coroutine behaves as expected.
        cr = self.func()
        # Initialise the coroutine.
        _ = cr.send(None)
        self.assertEqual(cr.send("spam"), 1)
        self.assertEqual(cr.send("ham"), 2)
        self.assertEqual(cr.send("eggs"), 3)
        self.assertEqual(cr.send("spam"), 4)

    def testFeed(self):
        # Test the feed() helper behaves as expected.
        cr = self.func()
        _ = cr.send(None)
        it = stats.co.feed(cr, "spam spam spam eggs bacon and spam".split())
        self.assertEqual(next(it), 1)
        self.assertEqual(next(it), 2)
        self.assertEqual(next(it), 3)
        self.assertEqual(next(it), 4)
        self.assertEqual(next(it), 5)
        self.assertEqual(next(it), 6)
        self.assertEqual(next(it), 7)
        self.assertRaises(StopIteration, next, it)
        self.assertRaises(StopIteration, next, it)


class CoSumTest(unittest.TestCase, TestConsumerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.co.sum

    def testAlias(self):
        # stats.co.sum is documented as an alias (otherwise we would have
        # to modify the docstring to hide the fact, and that's a PITA). So
        # test for it here.
        self.assertTrue(self.func is stats.running_sum)

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


class CoMeanTest(unittest.TestCase, TestConsumerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.co.mean

    def testMean(self):
        cr = self.func()
        self.assertEqual(cr.send(7), 7.0)
        self.assertEqual(cr.send(3), 5.0)
        self.assertEqual(cr.send(5), 5.0)
        self.assertEqual(cr.send(-5), 2.5)
        self.assertEqual(cr.send(0), 2.0)
        self.assertEqual(cr.send(9.5), 3.25)


class CoEWMATest(unittest.TestCase, TestConsumerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.co.ewma

    def testAverages(self):
        # Test the calculated averages.
        cr = self.func()
        self.assertEqual(cr.send(64), 64.0)
        self.assertEqual(cr.send(32), 48.0)
        self.assertEqual(cr.send(16), 32.0)
        self.assertEqual(cr.send(8), 20.0)
        self.assertEqual(cr.send(4), 12.0)
        self.assertEqual(cr.send(2), 7.0)
        self.assertEqual(cr.send(1), 4.0)

    def testAveragesAlpha(self):
        # Test the calculated averages with a specified alpha.
        cr = self.func(0.75)
        self.assertEqual(cr.send(64), 64.0)
        self.assertEqual(cr.send(32), 40.0)
        self.assertEqual(cr.send(58), 53.5)
        self.assertEqual(cr.send(48), 49.375)

    def testBadAlpha(self):
        # Test behaviour with an invalid alpha.
        for a in (None, 'spam', [1], (2,3), {}):
            self.assertRaises(stats.StatsError, self.func, a)


class CoWelfordTest(unittest.TestCase, TestConsumerMixin):
    # Test private _welford function.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.co._welford

    def test_welford(self):
        cr = self.func()
        # Expected results calculated by hand, then confirmed using this
        # RPL program: « Σ+ PVAR NΣ * »
        self.assertEqual(cr.send(2), (1, 0.0))
        self.assertEqual(cr.send(3), (2, 0.5))
        self.assertEqual(cr.send(4), (3, 2.0))
        self.assertEqual(cr.send(5), (4, 5.0))
        self.assertEqual(cr.send(6), (5, 10.0))
        cr = self.func()
        # Here I got lazy, and didn't bother with the hand calculations :)
        self.assertEqual(cr.send(3), (1, 0.0))
        self.assertEqual(cr.send(5), (2, 2.0))
        self.assertEqual(cr.send(4), (3, 2.0))
        self.assertEqual(cr.send(3), (4, 2.75))
        self.assertEqual(cr.send(5), (5, 4.0))
        self.assertEqual(cr.send(4), (6, 4.0))
        t = cr.send(-2)
        t = (t[0], round(t[1], 10))
        self.assertEqual(t, (7, 34.8571428571))


class CoPVarTest(test_stats.NumericTestCase, TestConsumerMixin):
    # Test coroutine population variance.
    tol = 2e-7
    rel = 2e-7

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.co.pvariance
        self.data = [2, 3, 5, 1, 3.5]
        self.expected = [0.0, 0.25, 14/9, 2.1875, 1.84]

    def testMain(self):
        cr = self.func()
        for x, expected in zip(self.data, self.expected):
            self.assertApproxEqual(cr.send(x), expected, tol=3e-16, rel=None)

    def testShift(self):
        cr1 = self.func()
        data1 = [random.gauss(3.5, 2.5) for _ in range(50)]
        expected = list(stats.co.feed(cr1, data1))
        cr2 = self.func()
        data2 = [x + 1e9 for x in data1]
        result = list(stats.co.feed(cr2, data2))
        self._compare_lists(result, expected)

    def _compare_lists(self, actual, expected):
        assert len(actual) == len(expected)
        for a,e in zip(actual, expected):
            if math.isnan(a) and math.isnan(e):
                self.assertTrue(True)
            else:
                self.assertApproxEqual(a, e)


class CoPstdevTest(CoPVarTest):
    # Test coroutine population std dev.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.co.pstdev
        self.expected = [math.sqrt(x) for x in self.expected]


class CoVarTest(CoPVarTest):
    # Test coroutine sample variance.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.co.variance
        n = len(self.data)
        self.first = self.data[0]
        del self.data[0]
        self.expected = [x*i/(i-1) for i,x in enumerate(self.expected[1:], 2)]

    def testMain(self):
        cr = self.func()
        x = cr.send(self.first)
        self.assertTrue(math.isnan(x), 'expected nan but got %r' % x)
        for x, expected in zip(self.data, self.expected):
            self.assertApproxEqual(cr.send(x), expected)


class CoStdevTest(CoVarTest):
    # Test coroutine sample std dev.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.co.stdev
        self.expected = [math.sqrt(x) for x in self.expected]


class CoCorrTest(test_stats.NumericTestCase):
    tol = 1e-14

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.co.corr

    def make_data(self, n):
        """Return n pairs of data."""
        def rand():
            return random.uniform(-0.5, 0.5)
        def f(x):
            return (2.3+rand())*x - (0.3+rand())
        domain = range(-17, -17+3*n, 3)
        assert len(domain) == n
        data = [(x, f(x)) for x in domain]
        random.shuffle(data)
        return data

    def get_final_result(self, values):
        cr = self.func()
        for xy in values:
            result = cr.send(xy)
        return result

    def testOrder(self):
        # The order that data is presented shouldn't matter to the
        # final result (although intermediate results may differ).
        xydata = self.make_data(100)
        a = self.get_final_result(xydata)
        random.shuffle(xydata)
        b = self.get_final_result(xydata)
        self.assertApproxEqual(a, b, tol=1e-14)

    def testFirstNan(self):
        # Test that the first result is always a NAN.
        for x in (-11.5, -2, 0, 0.25, 17, 45.95, 1e120):
            for y in (-8.5, -2, 0, 0.5, 31.35, 1e99):
                cr = self.func()
                self.assertTrue(math.isnan(cr.send((x,y))))

    def testPerfectCorrelation(self):
        xydata = [(x, 2.3*x - 0.8) for x in range(-17, 291, 3)]
        random.shuffle(xydata)
        cr = self.func()
        # Skip the equality test on the first value.
        xydata = iter(xydata)
        cr.send(next(xydata))
        for xy in xydata:
            self.assertApproxEqual(cr.send(xy), 1.0, tol=1e-15)

    def testPerfectAntiCorrelation(self):
        xydata = [(x, 273.4 - 3.1*x) for x in range(-22, 654, 7)]
        random.shuffle(xydata)
        cr = self.func()
        # Skip the equality test on the first value.
        xydata = iter(xydata)
        cr.send(next(xydata))
        for xy in xydata:
            self.assertApproxEqual(cr.send(xy), -1.0, tol=1e-15)

    def testPerfectZeroCorrelation(self):
        data = [(x, y) for x in range(1, 10) for y in range(1, 10)]
        result = self.get_final_result(data)
        self.assertApproxEqual(result, 0.0, tol=1e-15)

    def testExact(self):
        xdata = [0, 10, 4, 8, 8]
        ydata = [2, 6, 2, 4, 6]
        result = self.get_final_result(zip(xdata, ydata))
        self.assertEqual(result, 28/32)

    def testDuplicate(self):
        # corr shouldn't change if you duplicate each point.
        # Try first with a high correlation.
        xdata = [random.uniform(-5, 15) for _ in range(15)]
        ydata = [x - 0.5 + random.random() for x in xdata]
        xydata = list(zip(xdata, ydata))
        a = self.get_final_result(xydata)
        b = self.get_final_result(xydata*2)
        self.assertApproxEqual(a, b)
        # And again with a (probably) low correlation.
        ydata = [random.uniform(-5, 15) for _ in range(15)]
        xydata = list(zip(xdata, ydata))
        a = self.get_final_result(xydata)
        b = self.get_final_result(xydata*2)
        self.assertApproxEqual(a, b)

    def testSameCoords(self):
        # Test correlation with (X,X) coordinate pairs.
        data = [random.uniform(-3, 5) for x in range(5)]  # Small list.
        result = self.get_final_result([(x, x) for x in data])
        self.assertApproxEqual(result, 1.0)
        data = [random.uniform(-30, 50) for x in range(100)]  # Medium.
        result = self.get_final_result([(x, x) for x in data])
        self.assertApproxEqual(result, 1.0)
        data = [random.uniform(-3000, 5000) for x in range(100000)]  # Large.
        result = self.get_final_result([(x, x) for x in data])
        self.assertApproxEqual(result, 1.0)

    def generate_stress_data(self, start, end, step):
        """Generate a wide range of X and Y data for stress-testing."""
        xfuncs = (lambda x: x,
                  lambda x: 12345*x + 9876,
                  lambda x: 1e9*x,
                  lambda x: 1e-9*x,
                  lambda x: 1e-7*x + 3,
                  lambda x: 846*x - 423,
                  )
        yfuncs = (lambda y: y,
                  lambda y: 67890*y + 6428,
                  lambda y: 1e9*y,
                  lambda y: 1e-9*y,
                  lambda y: 2342*y - 1171,
                  )
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
        for xdata, ydata in self.generate_stress_data(5, 351, 23):
            cr = self.func()
            xydata = zip(xdata, ydata)
            # Skip the first value, as it is a NAN.
            cr.send(next(xydata))
            for xy in xydata:
                self.assertTrue(-1.0 <= cr.send(xy) <= 1.0)

    def testShift(self):
        # Shifting the data by a constant amount shouldn't change the
        # correlation. In practice, it may introduce some error. We allow
        # for that by increasing the tolerance as the shift gets bigger.
        xydata = self.make_data(100)
        a = self.get_final_result(xydata)
        offsets = [(42, -99), (1.2e6, 4.5e5), (7.8e9, 3.6e9)]
        tolerances = [self.tol, 5e-10, 1e-6]
        for (x0,y0), tol in zip(offsets, tolerances):
            data = [(x+x0, y+y0) for x,y in xydata]
            b = self.get_final_result(data)
            self.assertApproxEqual(a, b, tol=tol)


class CoCalcRTest(unittest.TestCase):
    # Test the _calc_r private function.

    def testNan(self):
        # _calc_r should return a NAN if either of the X or Y args are zero.
        result = stats.co._calc_r(0, 1, 2)
        self.assertTrue(math.isnan(result))
        result = stats.co._calc_r(1, 0, 2)
        self.assertTrue(math.isnan(result))

    def testAssertionFails(self):
        # _calc_r should include an assertion. Engineer a failure of it.
        if __debug__:
            self.assertRaises(AssertionError, stats.co._calc_r, 10, 10, 11)
            self.assertRaises(AssertionError, stats.co._calc_r, 10, 10, -11)

    def testMain(self):
        self.assertEqual(stats.co._calc_r(25, 36, 15), 0.5)



# -- multivar module --------------------------------------------------

class MultivarGlobalsTest(test_stats.GlobalsTest):
    module = stats.multivar


# -- order module -----------------------------------------------------

class OrderGlobalsTest(test_stats.GlobalsTest):
    module = stats.order


# -- univar module ----------------------------------------------------

class UnivarGlobalsTest(test_stats.GlobalsTest):
    module = stats.univar



# === Run tests ===

def test_main():
    unittest.main()


if __name__ == '__main__':
    test_main()

