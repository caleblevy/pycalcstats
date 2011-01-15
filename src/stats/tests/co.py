#!/usr/bin/env python3

##  Copyright (c) 2011 Steven D'Aprano.
##  See the file stats/__init__.py for the licence terms for this software.

"""
Test suite for the stats.co module.

"""

import inspect
import math
import random
import unittest

from stats.tests import NumericTestCase
import stats.tests.common as common

# The module to be tested:
import stats.co


# Helper mixin classes.

class TestConsumerMixin:
    def testIsConsumer(self):
        # Test that the function is a consumer.
        cr = self.func()
        self.assertTrue(hasattr(cr, 'send'))


class FeedTest(unittest.TestCase, TestConsumerMixin):
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


class SumTest(unittest.TestCase, TestConsumerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.co.sum

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


class MeanTest(unittest.TestCase, TestConsumerMixin):
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


class WeightedRunningAverageTest(unittest.TestCase, TestConsumerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.co.weighted_running_average

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


class WelfordTest(unittest.TestCase, TestConsumerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.co._welford

    @unittest.skip('what?')
    def test_welford(self):
        cr = self.func()
        self.assertEqual(cr.send(2), (1, 0.0))
        self.assertEqual(cr.send(3), (2, 0.25))
        self.assertEqual(cr.send(4), (3, 1.25))
        self.assertEqual(cr.send(5), (4, 3.5))


class PVarianceTest(NumericTestCase, TestConsumerMixin):
    tol = 5e-7
    rel = 5e-8

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


class PstdevTest(PVarianceTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.co.pstdev
        self.expected = [math.sqrt(x) for x in self.expected]


class VarianceTest(PVarianceTest):
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


class StdevTest(VarianceTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.co.stdev
        self.expected = [math.sqrt(x) for x in self.expected]


"""
class CorrTest(NumericTestCase):
    # Common tests for corr() and corr1().
    # All calls to the test function must be the one-argument style.
    # See CorrExtrasTest for two-argument tests.

    HP_TEST_NAME = 'CORR'
    tol = 1e-14

    def __init__(self, *args, **kwargs):
        NumericTestCase.__init__(self, *args, **kwargs)
        self.func = stats.corr

    def testOrdered(self):
        # Order shouldn't matter.
        xydata = [(x, 2.7*x - 0.3) for x in range(-20, 30)]
        a = self.func(xydata)
        random.shuffle(xydata)
        b = self.func(xydata)
        self.assertEqual(a, b)

    def testPerfectCorrelation(self):
        xydata = [(x, 2.3*x - 0.8) for x in range(-17, 395, 3)]
        self.assertEqual(self.func(xydata), 1.0)

    def testPerfectAntiCorrelation(self):
        xydata = [(x, 273.4 - 3.1*x) for x in range(-22, 654, 7)]
        self.assertEqual(self.func(xydata), -1.0)

    def testPerfectZeroCorrelation(self):
        data = []
        for x in range(1, 10):
            for y in range(1, 10):
                data.append((x, y))
        self.assertEqual(self.func(data), 0)

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
        self.assertEqual(a, b)
        self.assertEqual(a, c)
        self.assertEqual(a, d)

    def testExact(self):
        xdata = [0, 10, 4, 8, 8]
        ydata = [2, 6, 2, 4, 6]
        self.assertEqual(self.func(zip(xdata, ydata)), 28/32)

    def testHP(self):
        # Compare against results calculated on a HP-48GX calculator.
        for i in range(NUM_HP_TESTS):
            record = hp_multivariate_test_data(i)
            result = self.func(record.DATA)
            expected = getattr(record, self.HP_TEST_NAME)
            self.assertApproxEqual(result, expected)

    def testDuplicate(self):
        # corr shouldn't change if you duplicate each point.
        # Try first with a high correlation.
        xdata = [random.uniform(-5, 15) for _ in range(15)]
        ydata = [x - 0.5 + random.random() for x in xdata]
        a = self.func(zip(xdata, ydata))
        b = self.func(zip(xdata*2, ydata*2))
        self.assertApproxEqual(a, b)
        # And again with a (probably) low correlation.
        ydata = [random.uniform(-5, 15) for _ in range(15)]
        a = self.func(zip(xdata, ydata))
        b = self.func(zip(xdata*2, ydata*2))
        self.assertApproxEqual(a, b)

    def testSame(self):
        data = [random.random() for x in range(5)]
        result = self.func([(x, x) for x in data])
        self.assertApproxEqual(result, 1.0)  # small list
        data = [random.random() for x in range(100)]
        result = self.func([(x, x) for x in data])
        self.assertApproxEqual(result, 1.0)  # medium list
        data = [random.random() for x in range(100000)]
        result = self.func([(x, x) for x in data])
        self.assertApproxEqual(result, 1.0)  # large list

    def generate_stress_data(self, start, end, step):
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
            result = self.func(zip(xdata, ydata))
            self.assertTrue(-1.0 <= result <= 1.0)

    def shifted_correlation(self, xdata, ydata, xdelta, ydelta):
        xdata = [x+xdelta for x in xdata]
        ydata = [y+ydelta for y in ydata]
        return self.func(zip(xdata, ydata))

    def testShift(self):
        # Shifting the data by a constant amount shouldn't change the
        # correlation.
        xdata = [random.random() for _ in range(50)]
        ydata = [random.random() for _ in range(50)]
        a = self.func(zip(xdata, ydata))
        offsets = [(42, -99), (1.2e6, 4.5e5), (7.8e9, 3.6e9)]
        tolerances = [self.tol, 5e-10, 1e-6]
        for (x0,y0), tol in zip(offsets, tolerances):
            b = self.shifted_correlation(xdata, ydata, x0, y0)
            self.assertApproxEqual(a, b, tol=tol)
class Corr1Test(CorrTest):
    def __init__(self, *args, **kwargs):
        CorrTest.__init__(self, *args, **kwargs)
        self.func = stats.corr1

    def testPerfectCorrelation(self):
        xydata = [(x, 2.3*x - 0.8) for x in range(-17, 395, 3)]
        self.assertApproxEqual(self.func(xydata), 1.0)

    def testPerfectZeroCorrelation(self):
        xydata = []
        for x in range(1, 10):
            for y in range(1, 10):
                xydata.append((x, y))
        self.assertApproxEqual(self.func(xydata), 0.0)

    def testOrdered(self):
        # Order shouldn't matter.
        xydata = [(x, 2.7*x - 0.3) for x in range(-20, 30)]
        a = self.func(xydata)
        random.shuffle(xydata)
        b = self.func(xydata)
        self.assertApproxEqual(a, b)

    def testStress(self):
        # Stress the corr1() function looking for failures of the
        # post-condition -1 <= r <= 1. We expect that there may be some,
        # (but hope there won't be!) so don't stop on the first error.
        failed = 0
        it = self.generate_stress_data(5, 358, 11)
        for count, (xdata, ydata) in enumerate(it, 1):
            result = self.func(zip(xdata, ydata))
            failed += not -1.0 <= result <= 1.0
        assert count == 33*6*5
        self.assertEqual(failed, 0,
            "%d out of %d out of range errors" % (failed, count))


"""
