#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##  Copyright (c) 2011 Steven D'Aprano.
##  See the file stats/__init__.py for the licence terms for this software.

# Note to self: copied in full the following tests:
# _test *.basic *.co *.common *.general *.order *.univar *.utils


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
import stats.co
import stats.multivar
import stats.order
import stats.univar
import stats.utils



# === Helper functions ===

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


class TestConsumerMixin:
    def testIsConsumer(self):
        # Test that the function is a consumer.
        cr = self.func()
        self.assertTrue(hasattr(cr, 'send'))


# === Unit tests ===

# Note: do not use self.fail... unit tests, as they are deprecated in
# Python 3.2. Although plural test cases such as self.testEquals and
# friends are not officially deprecated, they are discouraged.


USE_DEFAULT = object()
class NumericTestCase(unittest.TestCase):
    tol = None
    rel = 1e-9
    def assertApproxEqual(
        self, actual, expected, tol=USE_DEFAULT, rel=USE_DEFAULT, msg=None
        ):
        # Note that unlike many other unittest assert* methods, this
        # is asymmetric -- the first argument is treated differently from
        # the second.
        if tol is USE_DEFAULT: tol = self.tol
        if rel is USE_DEFAULT: rel = self.rel
        if (isinstance(actual, collections.Sequence) and
        isinstance(expected, collections.Sequence)):
            result = self._check_approx_seq(actual, expected, tol, rel, msg)
        else:
            result = self._check_approx_num(actual, expected, tol, rel, msg)
        if result:
            raise result

    def _check_approx_seq(self, actual, expected, tol, rel, msg):
        if len(actual) != len(expected):
            standardMsg = (
                "actual and expected sequences differ in length; expected"
                " %d items but found %d." % (len(expected), len(actual)))
            msg = self._formatMessage(msg, standardMsg)
            # DON'T raise the exception, return it to be raised later!
            return self.failureException(msg)
        for i, (a,e) in enumerate(zip(actual, expected)):
            result = self._check_approx_num(a, e, tol, rel, msg, i)
            if result is not None:
                return result

    def _check_approx_num(self, actual, expected, tol, rel, msg, idx=None):
        # Note that we reverse the order of the arguments.
        if approx_equal(expected, actual, tol, rel):
            # Test passes. Return early, we are done.
            return None
        # Otherwise we failed. Generate an exception and return it.
        standardMsg = self._make_std_err_msg(actual, expected, tol, rel, idx)
        msg = self._formatMessage(msg, standardMsg)
        # DON'T raise the exception, return it to be raised later!
        return self.failureException(msg)

    def _make_std_err_msg(self, actual, expected, tol, rel, idx):
        # Create the standard error message, starting with the common part,
        # which comes at the end.
        abs_err = abs(actual - expected)
        rel_err = abs_err/abs(expected) if expected else float('inf')
        err_msg = '    absolute error = %r\n    relative error = %r'
        # Now for the part that is not common to all messages.
        if idx is None:
            # Comparing two numeric values.
            idxheader = ''
        else:
            idxheader = 'numeric sequences first differs at index %d.\n' % idx
        if tol is rel is None:
            header = 'actual value %r is not equal to expected %r\n'
            items = (actual, expected, abs_err, rel_err)
        else:
            header = 'actual value %r differs from expected %r\n' \
                        '    by more than %s\n'
            t = []
            if tol is not None:
                t.append('tol=%r' % tol)
            if rel is not None:
                t.append('rel=%r' % rel)
            assert t
            items = (actual, expected, ' and '.join(t), abs_err, rel_err)
        standardMsg = (idxheader + header + err_msg) % items
        return standardMsg


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




class MinimalVarianceTest(NumericTestCase):
    # Minimal tests for variance and friends.

   def testVariance(self):
       data = [1, 2, 3]
       assert stats.mean(data) == 2
       self.assertEqual(stats.pvariance(data), 2/3)
       self.assertEqual(stats.variance(data), 1.0)
       self.assertEqual(stats.pstdev(data), math.sqrt(2/3))
       self.assertEqual(stats.stdev(data), 1.0)


class PVarianceTest(NumericTestCase, UnivariateMixin):
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


class VarianceTest(SingleDataFailMixin, PVarianceTest):

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




# FIXME Older tests waiting to be ported.
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
"""


# -- Test stats.multivar module --

class StatsMultivarGlobalsTest(unittest.TestCase, GlobalsMixin):
    module = stats.multivar


class QCorrTest(NumericTestCase, MultivariateMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.multivar.qcorr

    def testPerfectCorrelation(self):
        xdata = range(-42, 1100, 7)
        ydata = [3.5*x - 0.1 for x in xdata]
        self.assertEqual(self.func(zip(xdata, ydata)), 1.0)

    def testPerfectAntiCorrelation(self):
        xydata = [(1, 10), (2, 8), (3, 6), (4, 4), (5, 2)]
        self.assertEqual(self.func(xydata), -1.0)
        xdata = range(-23, 1000, 3)
        ydata = [875.1 - 4.2*x for x in xdata]
        self.assertEqual(self.func(zip(xdata, ydata)), -1.0)

    def testPerfectZeroCorrelation(self):
        data = []
        for x in range(1, 10):
            for y in range(1, 10):
                data.append((x, y))
        random.shuffle(data)
        self.assertEqual(self.func(data), 0)

    def testNan(self):
        # Vertical line:
        xdata = [1 for _ in range(50)]
        ydata = [random.random() for _ in range(50)]
        result = self.func(xdata, ydata)
        self.assertTrue(math.isnan(result))
        # Horizontal line:
        xdata = [random.random() for _ in range(50)]
        ydata = [1 for _ in range(50)]
        result = self.func(xdata, ydata)
        self.assertTrue(math.isnan(result))
        # Neither horizontal nor vertical:
        # Take x-values and y-values both = (1, 2, 2, 3) with median = 2.
        xydata = [(1, 2), (2, 3), (2, 1), (3, 2)]
        result = self.func(xydata)
        self.assertTrue(math.isnan(result))


class MultivariateSplitDecoratorTest(NumericTestCase):
    # Test that the multivariate split decorator works correctly.
    def get_split_result(self, *args):
        @stats.multivar._Multivariate.split_xydata
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
        @stats.multivar._Multivariate.merge_xydata
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
        result = stats.multivar._Multivariate.merge([])
        self.assertEqual(list(result), [])
        result = stats.multivar._Multivariate.merge([], [])
        self.assertEqual(list(result), [])

    def test_xy_together(self):
        xydata = [(1, 2), (3, 4), (5, 6)]
        expected = xydata[:]
        result = stats.multivar._Multivariate.merge(xydata)
        self.assertEqual(list(result), expected)

    def test_xy_apart(self):
        xdata = [1, 3, 5]
        ydata = [2, 4, 6]
        expected = list(zip(xdata, ydata))
        result = stats.multivar._Multivariate.merge(xdata, ydata)
        self.assertEqual(list(result), expected)

    def test_x_alone(self):
        xdata = [1, 3, 5]
        expected = list(zip(xdata, [None]*len(xdata)))
        result = stats.multivar._Multivariate.merge(xdata)
        self.assertEqual(list(result), expected)


class SplitTest(NumericTestCase):
    # Test _Multivariate split function independantly of the decorator.
    def test_empty(self):
        result = stats.multivar._Multivariate.split([])
        self.assertEqual(result, ([], []))
        result = stats.multivar._Multivariate.split([], [])
        self.assertEqual(result, ([], []))

    def test_xy_together(self):
        xydata = [(1, 2), (3, 4), (5, 6)]
        expected = ([1, 3, 5], [2, 4, 6])
        result = stats.multivar._Multivariate.split(xydata)
        self.assertEqual(result, expected)

    def test_xy_apart(self):
        xdata = [1, 3, 5]
        ydata = [2, 4, 6]
        result = stats.multivar._Multivariate.split(xdata, ydata)
        self.assertEqual(result, (xdata, ydata))

    def test_x_alone(self):
        xdata = [1, 3, 5]
        result = stats.multivar._Multivariate.split(xdata)
        self.assertEqual(result, (xdata, [None]*3))


class CorrTest(NumericTestCase, SingleDataFailMixin, MultivariateMixin):
    # All calls to the test function must be the one-argument style.
    # See CorrExtrasTest for two-argument tests.

    HP_TEST_NAME = 'CORR'
    tol = 1e-14

    def __init__(self, *args, **kwargs):
        NumericTestCase.__init__(self, *args, **kwargs)
        self.func = stats.multivar.corr

    def testPerfectCorrelation(self):
        xdata = [0.0, 0.1, 0.25, 1.2, 1.75]
        ydata = [2.5*x + 0.3 for x in xdata]
        self.assertAlmostEqual(self.func(zip(xdata, ydata)), 1.0, places=14)
        xydata = [(x, 2.3*x - 0.8) for x in range(-17, 395, 3)]
        self.assertEqual(self.func(xydata), 1.0)

    def testPerfectAntiCorrelation(self):
        xdata = [0.0, 0.1, 0.25, 1.2, 1.75]
        ydata = [9.7 - 2.5*x for x in xdata]
        self.assertAlmostEqual(self.func(zip(xdata, ydata)), -1.0, places=14)
        xydata = [(x, 273.4 - 3.1*x) for x in range(-22, 654, 7)]
        self.assertEqual(self.func(xydata), -1.0)

    def testPerfectZeroCorrelation(self):
        data = []
        for x in range(1, 10):
            for y in range(1, 10):
                data.append((x, y))
        self.assertEqual(self.func(data), 0)

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

    def stress_test2(self, xdata, ydata):
        xfuncs = (lambda x: -1.2345e7*x - 23.42, lambda x: 9.42e-6*x + 2.1)
        yfuncs = (lambda y: -2.9234e7*y + 1.97, lambda y: 7.82e8*y - 307.9)
        for fx, fy in [(fx,fy) for fx in xfuncs for fy in yfuncs]:
            xs = [fx(x) for x in xdata]
            ys = [fy(y) for y in ydata]
            result = self.func(zip(xs, ys))
            self.assertTrue(-1.0 <= result <= 1.0)

    def testStress(self):
        # Stress the corr() function looking for failures of the
        # post-condition -1 <= r <= 1.
        for xdata, ydata in self.generate_stress_data(5, 351, 23):
            result = self.func(zip(xdata, ydata))
            self.assertTrue(-1.0 <= result <= 1.0)
        # A few extra stress tests.
        for i in range(6, 22, 3):
            xdata = [random.uniform(-100, 300) for _ in range(i)]
            ydata = [random.uniform(-5000, 5000) for _ in range(i)]
            self.stress_test2(xdata, ydata)

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

"""
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

class PCovTest(NumericTestCase, SingleDataFailMixin, MultivariateMixin):
    HP_TEST_NAME = 'PCOV'
    tol = 5e-12
    rel = 1e-8

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.multivar.pcov

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
        super().__init__(*args, **kwargs)
        self.func = stats.multivar.cov

    def testReduce(self):
        # Covariance reduces to variance if X == Y.
        data = [random.random() for _ in range(50)]
        a = stats.variance(data)
        b = self.func(zip(data, data))
        self.assertApproxEqual(a, b)


class LinrTest(NumericTestCase):
    HP_TEST_NAME = 'LINFIT'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.multivar.linr

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


# -- Test stats.order module --




# -- Test stats.univar module --



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



# -- Test stats.utils module --

class StatsUtilsGlobalsTest(unittest.TestCase, GlobalsMixin):
    module = stats.utils


class AsSequenceTest(unittest.TestCase):
    def testIdentity(self):
        data = [1, 2, 3]
        self.assertTrue(stats.utils.as_sequence(data) is data)
        data = tuple(data)
        self.assertTrue(stats.utils.as_sequence(data) is data)

    def testSubclass(self):
        def make_subclass(kind):
            # Helper function to make a subclass from the given class.
            class Subclass(kind):
                pass
            return Subclass

        for cls in (tuple, list):
            subcls = make_subclass(cls)
            data = subcls([1, 2, 3])
            assert type(data) is not cls
            assert issubclass(type(data), cls)
            self.assertTrue(stats.utils.as_sequence(data) is data)

    def testOther(self):
        data = range(20)
        assert type(data) is not list
        result = stats.utils.as_sequence(data)
        self.assertEqual(result, list(data))
        self.assertTrue(isinstance(result, list))


class ValidateIntTest(unittest.TestCase):
    def testIntegers(self):
        for n in (-2**100, -100, -1, 0, 1, 23, 42, 2**80, 2**100):
            self.assertIsNone(stats.utils._validate_int(n))

    def testSubclasses(self):
        class MyInt(int):
            pass
        for n in (True, False, MyInt(), MyInt(-101), MyInt(123)):
            self.assertIsNone(stats.utils._validate_int(n))

    def testGoodFloats(self):
        for n in (-100.0, -1.0, 0.0, 1.0, 23.0, 42.0, 1.23456e18):
            self.assertIsNone(stats.utils._validate_int(n))

    def testBadFloats(self):
        for x in (-100.1, -1.2, 0.3, 1.4, 23.5, 42.6, float('nan')):
            self.assertRaises(ValueError, stats.utils._validate_int, x)

    def testBadInfinity(self):
        for x in (float('-inf'), float('inf')):
            self.assertRaises(OverflowError, stats.utils._validate_int, x)

    def testBadTypes(self):
        for obj in ("a", "1", [], {}, object(), None):
            self.assertRaises((ValueError, TypeError),
                stats.utils._validate_int, obj)


class RoundTest(unittest.TestCase):
    UP = stats.utils._UP
    DOWN = stats.utils._DOWN
    EVEN = stats.utils._EVEN

    def testRoundDown(self):
        f = stats.utils._round
        self.assertEqual(f(1.4, self.DOWN), 1)
        self.assertEqual(f(1.5, self.DOWN), 1)
        self.assertEqual(f(1.6, self.DOWN), 2)
        self.assertEqual(f(2.4, self.DOWN), 2)
        self.assertEqual(f(2.5, self.DOWN), 2)
        self.assertEqual(f(2.6, self.DOWN), 3)

    def testRoundUp(self):
        f = stats.utils._round
        self.assertEqual(f(1.4, self.UP), 1)
        self.assertEqual(f(1.5, self.UP), 2)
        self.assertEqual(f(1.6, self.UP), 2)
        self.assertEqual(f(2.4, self.UP), 2)
        self.assertEqual(f(2.5, self.UP), 3)
        self.assertEqual(f(2.6, self.UP), 3)

    def testRoundEven(self):
        f = stats.utils._round
        self.assertEqual(f(1.4, self.EVEN), 1)
        self.assertEqual(f(1.5, self.EVEN), 2)
        self.assertEqual(f(1.6, self.EVEN), 2)
        self.assertEqual(f(2.4, self.EVEN), 2)
        self.assertEqual(f(2.5, self.EVEN), 2)
        self.assertEqual(f(2.6, self.EVEN), 3)


class SortedDataDecoratorTest(unittest.TestCase):
    # Test that the sorted_data decorator works correctly.
    def testDecorator(self):
        @stats.utils.sorted_data
        def f(data):
            return data

        values = random.sample(range(1000), 100)
        sorted_values = sorted(values)
        while values == sorted_values:
            # Ensure values aren't sorted.
            random.shuffle(values)
        result = f(values)
        self.assertNotEqual(result, values)
        self.assertEqual(result, sorted_values)

