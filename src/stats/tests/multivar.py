#!/usr/bin/env python3

##  Copyright (c) 2011 Steven D'Aprano.
##  See the file stats/__init__.py for the licence terms for this software.

"""
Test suite for the stats.multivar module.

"""

import collections

from stats.tests import NumericTestCase
import stats.tests.common as common

# The module to be tested:
import stats.multivar


# === Helper functions ===

NUM_HP_TESTS = 3
def hp_multivariate_test_data(switch):
    """Generate test data to match results calculated on the HP-48GX."""
    record = collections.namedtuple('record', 'DATA CORR COV PCOV LINFIT')
    if switch == 0:
        # Equivalent to this RPL code:
        # « CLΣ DEG 30 200 FOR X X X SIN →V2 Σ+ NEXT »
        xdata = range(30, 201)
        ydata = [math.sin(math.radians(x)) for x in xdata]
        assert len(xdata) == len(ydata) == 171
        assert sum(xdata) == 19665
        assert round(sum(ydata), 9) == 103.536385403
        DATA = zip(xdata, ydata)
        CORR = -0.746144846212
        COV = -14.3604967839
        PCOV = -14.2765172706
        LINFIT = (1.27926505682, -5.85903581555e-3)
    elif switch == 1:
        # Equivalent to this RPL code:
        # « CLΣ -5 15 FOR X X 2 X - X SQ + →V2 Σ+ .1 STEP »
        xdata = [i/10 for i in range(-50, 151)]
        ydata = [x**2 - x + 2 for x in xdata]
        assert len(xdata) == len(ydata) == 201
        assert round(sum(xdata), 11) == 1005
        assert round(sum(ydata), 11) == 11189
        DATA = zip(xdata, ydata)
        CORR = 0.866300845681
        COV = 304.515
        PCOV = 303
        LINFIT = (10 + 2/3, 9)
    elif switch == 2:
        # Equivalent to this RPL code:
        # « CLΣ -30 60 FOR I I 3 / 500 I + √ →V2 Σ+ NEXT »
        xdata = [i/3 for i in range(-30, 61)]
        ydata = [math.sqrt(500 + i) for i in range(-30, 61)]
        assert len(xdata) == len(ydata) == 91
        assert round(sum(xdata), 11) == 455
        assert round(sum(ydata), 6) == round(2064.4460877, 6)
        DATA = zip(xdata, ydata)
        CORR = 0.999934761605
        COV = 5.1268171707
        PCOV = 5.07047852047
        LINFIT = (22.3555373622, 6.61366763539e-2)
    return record(DATA, CORR, COV, PCOV, LINFIT)


# === Test multivariate statistics ===

"""

class MultivariateSplitDecoratorTest(NumericTestCase):
    # Test that the multivariate split decorator works correctly.
    def get_split_result(self, *args):
        @stats._Multivariate.split_xydata
        def f(xdata, ydata):
            return (xdata, ydata)
        return f(*args)

    def test_empty(self):
        empty = iter([])
        result = self.get_split_result(empty)
        self.assertEqual(result, ([], []))
        result = self.get_split_result(empty, empty)
        self.assertEqual(result, ([], []))

    def test_xy_apart(self):
        xdata = range(8)
        ydata = [2**i for i in xdata]
        result = self.get_split_result(xdata, ydata)
        self.assertEqual(result, (list(xdata), ydata))

    def test_xy_together(self):
        xydata = [(i, 2**i) for i in range(8)]
        xdata = [x for x,y in xydata]
        ydata = [y for x,y in xydata]
        result = self.get_split_result(xydata)
        self.assertEqual(result, (xdata, ydata))

    def test_x_alone(self):
        xdata = [2, 4, 6, 8]
        result = self.get_split_result(xdata)
        self.assertEqual(result, (xdata, [None]*4))


class MultivariateMergeDecoratorTest(NumericTestCase):
    # Test that the multivariate merge decorator works correctly.
    def get_merge_result(self, *args):
        @stats._Multivariate.merge_xydata
        def f(xydata):
            return list(xydata)
        return f(*args)

    def test_empty(self):
        empty = iter([])
        result = self.get_merge_result(empty)
        self.assertEqual(result, [])
        result = self.get_merge_result(empty, empty)
        self.assertEqual(result, [])

    def test_xy_apart(self):
        expected = [(i, 2**i) for i in range(8)]
        xdata = [x for (x,y) in expected]
        ydata = [y for (x,y) in expected]
        result = self.get_merge_result(xdata, ydata)
        self.assertEqual(result, expected)

    def test_xy_together(self):
        expected = [(i, 2**i) for i in range(8)]
        xdata = [x for x,y in expected]
        ydata = [y for x,y in expected]
        result = self.get_merge_result(zip(xdata, ydata))
        self.assertEqual(result, expected)

    def test_x_alone(self):
        xdata = [2, 4, 6, 8]
        expected = [(x, None) for x in xdata]
        result = self.get_merge_result(xdata)
        self.assertEqual(result, expected)


class MergeTest(NumericTestCase):
    # Test _Multivariate merge function independantly of the decorator.
    def test_empty(self):
        result = stats._Multivariate.merge([])
        self.assertEqual(list(result), [])
        result = stats._Multivariate.merge([], [])
        self.assertEqual(list(result), [])

    def test_xy_together(self):
        xydata = [(1, 2), (3, 4), (5, 6)]
        expected = xydata[:]
        result = stats._Multivariate.merge(xydata)
        self.assertEqual(list(result), expected)

    def test_xy_apart(self):
        xdata = [1, 3, 5]
        ydata = [2, 4, 6]
        expected = list(zip(xdata, ydata))
        result = stats._Multivariate.merge(xdata, ydata)
        self.assertEqual(list(result), expected)

    def test_x_alone(self):
        xdata = [1, 3, 5]
        expected = list(zip(xdata, [None]*len(xdata)))
        result = stats._Multivariate.merge(xdata)
        self.assertEqual(list(result), expected)


class SplitTest(NumericTestCase):
    # Test _Multivariate split function independantly of the decorator.
    def test_empty(self):
        result = stats._Multivariate.split([])
        self.assertEqual(result, ([], []))
        result = stats._Multivariate.split([], [])
        self.assertEqual(result, ([], []))

    def test_xy_together(self):
        xydata = [(1, 2), (3, 4), (5, 6)]
        expected = ([1, 3, 5], [2, 4, 6])
        result = stats._Multivariate.split(xydata)
        self.assertEqual(result, expected)

    def test_xy_apart(self):
        xdata = [1, 3, 5]
        ydata = [2, 4, 6]
        result = stats._Multivariate.split(xdata, ydata)
        self.assertEqual(result, (xdata, ydata))

    def test_x_alone(self):
        xdata = [1, 3, 5]
        result = stats._Multivariate.split(xdata)
        self.assertEqual(result, (xdata, [None]*3))





class QCorrTest(NumericTestCase):
    def testPerfectCorrelation(self):
        xdata = range(-42, 1100, 7)
        ydata = [3.5*x - 0.1 for x in xdata]
        self.assertEqual(stats.qcorr(xdata, ydata), 1.0)

    def testPerfectAntiCorrelation(self):
        xydata = [(1, 10), (2, 8), (3, 6), (4, 4), (5, 2)]
        self.assertEqual(stats.qcorr(xydata), -1.0)
        xdata = range(-23, 1000, 3)
        ydata = [875.1 - 4.2*x for x in xdata]
        self.assertEqual(stats.qcorr(xdata, ydata), -1.0)

    def testPerfectZeroCorrelation(self):
        data = []
        for x in range(1, 10):
            for y in range(1, 10):
                data.append((x, y))
        random.shuffle(data)
        self.assertEqual(stats.qcorr(data), 0)

    def testCompareAlternateInput(self):
        # Compare the xydata vs. xdata, ydata input arguments.
        xdata = [random.random() for _ in range(1000)]
        ydata = [random.random() for _ in range(1000)]
        a = stats.qcorr(xdata, ydata)
        b = stats.qcorr(list(zip(xdata, ydata)))
        self.assertEqual(a, b)

    def testNan(self):
        # Vertical line:
        xdata = [1 for _ in range(50)]
        ydata = [random.random() for _ in range(50)]
        result = stats.qcorr(xdata, ydata)
        self.assertTrue(math.isnan(result))
        # Horizontal line:
        xdata = [random.random() for _ in range(50)]
        ydata = [1 for _ in range(50)]
        result = stats.qcorr(xdata, ydata)
        self.assertTrue(math.isnan(result))
        # Neither horizontal nor vertical:
        # Take x-values and y-values both = (1, 2, 2, 3) with median = 2.
        xydata = [(1, 2), (2, 3), (2, 1), (3, 2)]
        result = stats.qcorr(xydata)
        self.assertTrue(math.isnan(result))

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
        self.assertEqual(a, b)
        self.assertEqual(a, c)
        self.assertEqual(a, d)


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


class CorrExtrasTest(NumericTestCase):
    def __init__(self, *args, **kwargs):
        NumericTestCase.__init__(self, *args, **kwargs)
        self.func = stats.corr

    def testSimple(self):
        # Simple test originally from a doctest in _corr2.
        xdata = [0.0, 0.1, 0.25, 1.2, 1.75]
        ydata = [2.5*x + 0.3 for x in xdata]  # Perfect correlation.
        self.assertAlmostEqual(self.func(xdata, ydata), 1.0, places=14)
        ydata = [10-y for y in ydata]
        self.assertAlmostEqual(self.func(xdata, ydata), -1.0, places=14)

    def testCompareAlternateInput(self):
        # Compare the xydata vs. xdata, ydata input arguments.
        xdata = [random.random() for _ in range(1000)]
        ydata = [random.random() for _ in range(1000)]
        a = self.func(xdata, ydata)
        b = self.func(zip(xdata, ydata))
        self.assertEqual(a, b)

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
        self.assertEqual(a, b)
        self.assertEqual(a, c)

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


class PCovTest(NumericTestCase):
    HP_TEST_NAME = 'PCOV'
    tol = 5e-12
    rel = 1e-8

    def __init__(self, *args, **kwargs):
        NumericTestCase.__init__(self, *args, **kwargs)
        self.func = stats.pcov

    def testEmpty(self):
        self.assertRaises(ValueError, self.func, [])

    def testSingleton(self):
        self.assertEqual(self.func([(1, 2)]), 0.0)

    def testSymmetry(self):
        data1 = [random.random() for _ in range(10)]
        data2 = [random.random() for _ in range(10)]
        a = self.func(zip(data1, data2))
        b = self.func(zip(data2, data1))
        self.assertEqual(a, b)

    def testEqualPoints(self):
        # Equal X values.
        data = [(23, random.random()) for _ in range(50)]
        self.assertEqual(self.func(data), 0.0)
        # Equal Y values.
        data = [(random.random(), 42) for _ in range(50)]
        self.assertEqual(self.func(data), 0.0)
        # Both equal.
        data = [(23, 42)]*50
        self.assertEqual(self.func(data), 0.0)

    def testReduce(self):
        # Covariance reduces to variance if X == Y.
        data = [random.random() for _ in range(50)]
        a = stats.pvariance(data)
        b = self.func(zip(data, data))
        self.assertApproxEqual(a, b)

    def testShift(self):
        xdata = [random.random() for _ in range(50)]
        ydata = [random.random() for _ in range(50)]
        a = self.func(zip(xdata, ydata))
        for x0, y0 in [(-23, 89), (193, -4362), (3.7e5, 2.9e6)]:
            xdata = [x+x0 for x in xdata]
            ydata = [y+y0 for y in ydata]
            b = self.func(zip(xdata, ydata))
            self.assertApproxEqual(a, b)
        for x0, y0 in [(1.4e9, 8.1e9), (-2.3e9, 5.8e9)]:
            xdata = [x+x0 for x in xdata]
            ydata = [y+y0 for y in ydata]
            b = self.func(zip(xdata, ydata))
            self.assertApproxEqual(a, b, tol=1e-7)

    def testHP(self):
        # Compare against results calculated on a HP-48GX calculator.
        for i in range(NUM_HP_TESTS):
            record = hp_multivariate_test_data(i)
            result = self.func(record.DATA)
            exp = getattr(record, self.HP_TEST_NAME)
            self.assertApproxEqual(result, exp)


class CovTest(PCovTest):
    HP_TEST_NAME = 'COV'

    def __init__(self, *args, **kwargs):
        PCovTest.__init__(self, *args, **kwargs)
        self.func = stats.cov

    def testSingleton(self):
        self.assertRaises(ValueError, self.func, [(1, 2)])

    def testReduce(self):
        # Covariance reduces to variance if X == Y.
        data = [random.random() for _ in range(50)]
        a = stats.variance(data)
        b = self.func(zip(data, data))
        self.assertApproxEqual(a, b)


class ErrSumSqTest(NumericTestCase):

    @unittest.skip('not yet implemented')
    def testErrSumSq(self):
        self.fail()


class LinrTest(NumericTestCase):
    HP_TEST_NAME = 'LINFIT'

    def __init__(self, *args, **kwargs):
        NumericTestCase.__init__(self, *args, **kwargs)
        self.func = stats.linr

    def testTwoTuple(self):
        # Test that linear regression returns a two tuple.
        data = [(1,2), (3, 5), (5, 9)]
        result = self.func(data)
        self.assertTrue(isinstance(result, tuple))
        self.assertTrue(len(result) == 2)

    def testHP(self):
        # Compare against results calculated on a HP-48GX calculator.
        for i in range(NUM_HP_TESTS):
            record = hp_multivariate_test_data(i)
            intercept, slope = self.func(record.DATA)
            a, b = getattr(record, self.HP_TEST_NAME)
            self.assertApproxEqual(intercept, a)
            self.assertApproxEqual(slope, b)

    def testEmpty(self):
        self.assertRaises(ValueError, self.func, [])

    def testSingleton(self):
        self.assertRaises(ValueError, self.func, [(1, 2)])


"""

