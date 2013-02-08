#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##  Copyright (c) 2011 Steven D'Aprano.
##  See the file stats/__init__.py for the licence terms for this software.

import collections
import functools
import inspect
import itertools
import math
import os
import pickle
import random
import unittest
import zipfile
# from test import support

# The module(s) to be tested:
import stats



# === Helper functions ===

def approx_equal(x, y, tol=1e-12, rel=1e-7):
    if tol is rel is None:
        # Fall back on exact equality.
        return x == y
    tests = []
    if tol is not None: tests.append(tol)
    if rel is not None: tests.append(rel*abs(x))
    assert tests
    return abs(x - y) <= max(tests)


def _get_extra_args(obj):
    try:
        extras = obj.extra_args
    except AttributeError:
        # By default, run the test once, with no extra arguments.
        extras = ((),)
    if not extras:
        raise RuntimeError('empty extras will disable tests')
    return extras


def handle_extra_args(func):
    # Decorate test methods so that they pass any extra positional arguments
    # specified in self.extra_args (if it exists). See the comment in the
    # UnivariateMixin test class for more detail.
    @functools.wraps(func)
    def inner(self, *args, **kwargs):
        for extra_args in _get_extra_args(self):
            a = args + tuple(extra_args)
            func(self, *a, **kwargs)
    return inner


def handle_data_sets(num_points):
    # Decorator factory returning a decorator which wraps its function
    # so as to run num_sets individual tests, with each test using num_points
    # individual data points. The method self.make_data is called with both
    # arguments to generate the data sets. See the UnivariateMixin class for
    # the default implementation.
    def decorator(func):
        @functools.wraps(func)
        def inner_handle_data_sets(self, *args, **kwargs):
            test_data = self.make_data(num_points)
            for data in test_data:
                func(self, list(data), *args, **kwargs)
        return inner_handle_data_sets
    return decorator


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


# === Mixin tests ===

class GlobalsMixin:
    # Test the state and/or existence of globals.
    expected_metadata = ["__doc__", "__all__"]

    def testMeta(self):
        # Test for the existence of metadata.
        for meta in self.expected_metadata:
            self.assertTrue(hasattr(self.module, meta),
                            "%s not present" % meta)

    def testCheckAll(self):
        # Check everything in __all__ exists.
        module = self.module
        for name in module.__all__:
            self.assertTrue(hasattr(module, name))
    # FIXME make sure that things that shouldn't be in __all__ aren't?


class UnivariateMixin:
# Common tests for most univariate functions that take a data argument.
#
# This tests the behaviour of functions of the form func(data [,...])
# without checking the value returned. Tests for correctness of the
# return value are not the responsibility of this class.
#
# Most functions won't care much about the length of the input data,
# provided there are sufficient data points (usually >= 1). But when
# testing the functions in stats.order, we do care about the length:
# we need to cover all four cases of len(data)%4 = 0, 1, 2, 3.

# This class has the following dependencies:
#
#   self.func       - The function being tested, assumed to take at
#                     least one argument.
#   self.extra_args - (optional) If it exists, a sequence of tuples to
#                     pass to the test function as extra positional
#                     arguments.
#
# plus the assert* unittest methods.
#
# If the function needs no extra arguments, just don't define
# self.extra_args. Otherwise, calls to the test function may be made 1 or
# more times, using each tuple taken from self.extra_args. E.g. if
# self.extra_args = [(), (a,), (b,c)] then the function will be called three
# times per test:
#   self.func(data)
#   self.func(data, a)
#   self.func(data, b, c)
# (with data set appropriately by the test). This behaviour is enabled by
# the handle_extra_args decorator.

    # === Helper methods ===

    def make_data(self, num_points):
        """Return data sets of num_points elements each suitable for being
        passed to the test function. num_points should be a positive integer
        up to a maximum of 8, or None. If it is None, the data sets will
        have variable lengths.

        E.g. make_data(2) might return something like this:
            [ [1,2], [4,5], [6,7], ... ]

        (the actual number of data sets is an implementation detail, but
        will be at least 4) and the test function will be called:
            func([1, 2])
            func([4, 5])
            func([6, 7])
            ...

        This method is called by the handle_data_sets decorator.
        """

        # If num_points is None, we randomly select a mix of data lengths.
        # But not entirely at random -- we need to consider the stats.order
        # functions which take different paths depending on whether their
        # data argument has length 0, 1, 2, or 3 modulo 4. (Or in the case
        # of median, 0 or 1 modulo 2.) To ensure we cover all four of these
        # cases, we have to carefully choose our "random" mix.
        data = [  # 8 sets of data, each of length 8.
            [1, 2, 4, 8, 16, 32, 64, 128],
            [0.0, 0.25, 0.25, 1.5, 2.5, 2.5, 2.75, 4.75],
            [-0.75, 0.75, 1.5, 2.25, 3.25, 4.5, 5.75, 6.0],
            [925.0, 929.5, 934.25, 940.0, 941.25, 941.25, 944.75, 946.25],
            [5.5, 2.75, 1.25, 0.0, -0.25, -0.5, -1.75, -2.0],
            [23.0, 23.5, 29.5, 31.25, 34.75, 42.0, 48.0, 52.25],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [-0.25, 1.75, 2.25, 3.5, 4.75, 5.5, 6.75, 7.25]
            ]  # For avoidance of rounding errors, prefer numbers which are
               # either ints or exact binary fractions.
        assert len(data) == 8
        assert all(len(d) == 8 for d in data)
        if num_points is None:
            # Cover the cases len(data)%4 -> 0...3
            for i in range(4):
                data[i] = data[i][:4+i]
                data[i+4] = data[i+4][:4+i]
            assert [len(d)%4 for d in data] == [0, 1, 2, 3]*2
        else:
            if num_points < 1:
                raise RuntimeError('too few test points, got %d' % num_points)
            n = min(num_points, 8)
            if n != 8:
                data = [d[:n] for d in data]
            assert [len(d) for d in data] == [n]*8
        assert len(data) == 8
        return data

    # === Test methods ===

    def testNoArgs(self):
        # Fail if given no arguments.
        self.assertRaises(TypeError, self.func)

    @handle_extra_args
    def testEmptyData(self, *args):
        # Fail when the data argument (first argument) is empty.
        for empty in ([], (), iter([])):
            self.assertRaises(ValueError, self.func, empty, *args)

    @handle_extra_args
    def testSingleData(self, *args):
        # Pass when the first argument has a single data point.
        for data in self.make_data(1):
            assert len(data) == 1
            _ = self.func(list(data), *args)

    @handle_extra_args
    def testDoubleData(self, *args):
        # Pass when the first argument has two data points.
        for x,y in self.make_data(2):
            _ = self.func([x,y], *args)

    @handle_extra_args
    def testTripleData(self, *args):
        # Pass when the first argument has three data points.
        for x,y,z in self.make_data(3):
            _ = self.func([x, y, z], *args)

    @handle_extra_args
    def testQuadPlusData(self, *args):
        # Pass when the first argument has four + data points.
        for n in range(4, 9):
            for t in self.make_data(n):
                _ = self.func(t, *args)

    @handle_data_sets(None)
    @handle_extra_args
    def testNoInPlaceModifications(self, data, *args):
        # Test that the function does not modify its input data.
        sorted_data = sorted(data)
        if len(data) > 1:  # Otherwise we loop forever.
            while data == sorted_data:
                random.shuffle(data)
        assert data != sorted(data)
        saved_data = data[:]
        assert data is not saved_data
        _ = self.func(data, *args)
        self.assertEqual(data, saved_data)

    @handle_data_sets(None)
    @handle_extra_args
    def testOrderDoesntMatter(self, data, *args):
        # Test that the result of the function shouldn't depend (much)
        # on the order of data points.
        data.sort()
        expected = self.func(data, *args)
        result = self.func(reversed(data), *args)
        self.assertEqual(expected, result)
        for i in range(10):
            random.shuffle(data)
            result = self.func(data, *args)
            self.assertApproxEqual(result, expected, tol=1e-13, rel=None)

    @handle_data_sets(None)
    @handle_extra_args
    def testDataTypeDoesntMatter(self, data, *args):
        # Test that the type of iterable data doesn't effect the result.
        expected = self.func(data, *args)
        class MyList(list):
            pass
        def generator(data):
            return (obj for obj in data)
        for kind in (list, tuple, iter, reversed, MyList, generator):
            result = self.func(kind(data), *args)
            self.assertApproxEqual(result, expected, tol=1e-13, rel=None)

    @handle_data_sets(None)
    @handle_extra_args
    def testFloatTypeDoesntMatter(self, data, *args):
        # Test that the type of float data shouldn't effect the result.
        expected = self.func(data, *args)
        class MyFloat(float):
            pass
        data = [MyFloat(x) for x in data]
        result = self.func(data, *args)
        self.assertEqual(expected, result)


class MultivariateMixin(UnivariateMixin):
    def make_data(self, num_points):
        data = super().make_data(num_points)
        # Now transform data like this:
        #   [ [x11, x12, x13, ...], [x21, x22, x23, ...], ... ]
        # into this:
        #   [ [(x11, 1), (x12, 2), (x13, 3), ...], ... ]
        for i in range(len(data)):
            d = data[i]
            d = [(x, j+1) for j,x in enumerate(d)]
            data[i] = d
        return data

    @handle_data_sets(None)
    @handle_extra_args
    def testFloatTypeDoesntMatter(self, data, *args):
        # Test that the type of float data shouldn't effect the result.
        expected = self.func(data, *args)
        class MyFloat(float):
            pass
        data = [tuple(map(MyFloat, t)) for t in data]
        result = self.func(data, *args)
        self.assertEqual(expected, result)


class SingleDataFailMixin:
    # Test that the test function fails with a single data point.
    # This class overrides the method with the same name in
    # UnivariateMixin.

    @handle_extra_args
    def testSingleData(self, *args):
        # Fail when given a single data point.
        for x in (1.0, 0.0, -2.5, 5.5):
            self.assertRaises(ValueError, self.func, [x], *args)


class DoubleDataFailMixin(SingleDataFailMixin):
    # Test that the test function fails with one or two data points.
    # This class overrides the methods with the same names in
    # UnivariateMixin.

    @handle_extra_args
    def testDoubleData(self, *args):
        # Fail when the first argument is two data points.
        for x, y in ((1.0, 0.0), (-2.5, 5.5), (2.3, 4.2)):
            self.assertRaises(ValueError, self.func, [x, y], *args)




# -- General tests not specific to any module --

class CompareAgainstNumpyResultsTest(NumericTestCase):
    # Test the results we generate against some numpy equivalents.
    places = 8

    def __init__(self, *args, **kwargs):
        NumericTestCase.__init__(self, *args, **kwargs)
        # Read data from external test data file.
        # (In this case, produced by numpy and Python 2.5.)
        location = self.get_data_location('support/test_data.zip')
        # Now read the data from that file.
        zf = zipfile.ZipFile(location, 'r')
        self.data = pickle.loads(zf.read('data.pkl'))
        self.expected = pickle.loads(zf.read('results.pkl'))
        zf.close()

    def get_data_location(self, filename):
        # First we have to find our base location.
        import stats._tests
        location = os.path.split(stats._tests.__file__)[0]
        # Now add the filename to it.
        return os.path.join(location, filename)

    # FIXME assertAlmostEqual is not really the right way to do these
    # tests, as decimal places != significant figures.
    def testSum(self):
        result = stats.sum(self.data)
        expected = self.expected['sum']
        n = int(math.log(result, 10))  # Yuck.
        self.assertAlmostEqual(result, expected, places=self.places-n)

    def testProduct(self):
        result = stats.univar.product(self.data)
        expected = self.expected['product']
        self.assertAlmostEqual(result, expected, places=self.places)

    def testMean(self):
        result = stats.mean(self.data)
        expected = self.expected['mean']
        self.assertAlmostEqual(result, expected, places=self.places)

    def testRange(self):
        result = stats.order.range(self.data)
        expected = self.expected['range']
        self.assertAlmostEqual(result, expected, places=self.places)

    def testMidrange(self):
        result = stats.order.midrange(self.data)
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


class AssortedResultsTest(NumericTestCase):
    # Test some assorted statistical results against exact results
    # calculated by hand, and confirmed by HP-48GX calculations.
    places = 16

    def __init__(self, *args, **kwargs):
        NumericTestCase.__init__(self, *args, **kwargs)
        self.xdata = [1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 3/2, 5/2,
                      7/2, 9/2, 11/2, 13/2, 15/2, 17/2, 19/2]
        self.ydata = [1/4, 1/2, 3/2, 1, 1/2, 3/2, 1, 5/4, 5/2, 7/4,
                      9/4, 11/4, 11/4, 7/4, 13/4, 17/4]
        assert len(self.xdata) == len(self.ydata) == 16

    def testSums(self):
        Sx = stats.sum(self.xdata)
        Sy = stats.sum(self.ydata)
        self.assertAlmostEqual(Sx, 3295/64, places=self.places)
        self.assertAlmostEqual(Sy, 115/4, places=self.places)

    def testSumSqs(self):
        Sx2 = stats.sum(x**2 for x in self.xdata)
        Sy2 = stats.sum(x**2 for x in self.ydata)
        self.assertAlmostEqual(Sx2, 1366357/4096, places=self.places)
        self.assertAlmostEqual(Sy2, 1117/16, places=self.places)

    def testMeans(self):
        x = stats.mean(self.xdata)
        y = stats.mean(self.ydata)
        self.assertAlmostEqual(x, 3295/1024, places=self.places)
        self.assertAlmostEqual(y, 115/64, places=self.places)

    def testOtherSums(self):
        Sxx = stats.multivar.Sxx(zip(self.xdata, self.ydata))
        Syy = stats.multivar.Syy(zip(self.xdata, self.ydata))
        Sxy = stats.multivar.Sxy(zip(self.xdata, self.ydata))
        self.assertAlmostEqual(Sxx, 11004687/4096, places=self.places)
        self.assertAlmostEqual(Syy, 4647/16, places=self.places)
        self.assertAlmostEqual(Sxy, 197027/256, places=self.places)

    def testPVar(self):
        sx2 = stats.pvariance(self.xdata)
        sy2 = stats.pvariance(self.ydata)
        self.assertAlmostEqual(sx2, 11004687/1048576, places=self.places)
        self.assertAlmostEqual(sy2, 4647/4096, places=self.places)

    def testVar(self):
        sx2 = stats.variance(self.xdata)
        sy2 = stats.variance(self.ydata)
        self.assertAlmostEqual(sx2, 11004687/983040, places=self.places)
        self.assertAlmostEqual(sy2, 4647/3840, places=self.places)

    def testPCov(self):
        v = stats.multivar.pcov(self.xdata, self.ydata)
        self.assertAlmostEqual(v, 197027/65536, places=self.places)

    def testCov(self):
        v = stats.multivar.cov(self.xdata, self.ydata)
        self.assertAlmostEqual(v, 197027/61440, places=self.places)

    def testErrSumSq(self):
        se = stats.multivar.errsumsq(self.xdata, self.ydata)
        self.assertAlmostEqual(se, 96243295/308131236, places=self.places)

    def testLinr(self):
        a, b = stats.multivar.linr(self.xdata, self.ydata)
        expected_b = 3152432/11004687
        expected_a = 115/64 - expected_b*3295/1024
        self.assertAlmostEqual(a, expected_a, places=self.places)
        self.assertAlmostEqual(b, expected_b, places=self.places)

    def testCorr(self):
        r = stats.multivar.corr(zip(self.xdata, self.ydata))
        Sxx = 11004687/4096
        Syy = 4647/16
        Sxy = 197027/256
        expected = Sxy/math.sqrt(Sxx*Syy)
        self.assertAlmostEqual(r, expected, places=15)


# -- Test stats module --






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



