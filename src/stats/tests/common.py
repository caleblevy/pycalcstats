#!/usr/bin/env python3

##  Copyright (c) 2011 Steven D'Aprano.
##  See the file stats/__init__.py for the licence terms for this software.

"""
Common mixin tests for the stats package.

"""

import functools
import random


# === Helper functions ===

def _get_extra_args(obj):
    try:
        extras = obj.extras
    except AttributeError:
        # By default, run the test once, with no extra arguments.
        extras = ((),)
    if not extras:
        raise RuntimeError('empty extras will disable tests')
    return extras


def handle_extra_arguments(func):
    # Decorate test methods so that they pass any extra positional arguments
    # specified in self.extras (if it exists). See the comment in the
    # UnivariateMixin test class for more detail.
    @functools.wraps(func)
    def inner_handle_extra_args(self, *args, **kwargs):
        for extra_args in _get_extra_args(self):
            a = args + tuple(extra_args)
            func(self, *a, **kwargs)
    return inner_handle_extra_args


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
    #   self.func     - The function being tested, assumed to take at
    #                   least one argument.
    #   self.extras   - (optional) If it exists, a sequence of tuples to
    #                   pass to the test function as extra positional args.
    #
    # plus the assert* unittest methods.
    #
    # If the function needs no extra arguments, just don't define self.extras.
    # Otherwise, calls to the test function may be made 1 or more times,
    # using each tuple taken from self.extras.
    # e.g. if self.extras = [(), (a,), (b,c)] then the function may be
    # called three times per test:
    #   self.func(data)
    #   self.func(data, a)
    #   self.func(data, b, c)
    # (with data set appropriately by the test). This behaviour is
    # controlled by the handle_extra_arguments decorator.

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
        data = [
            [1, 2, 4, 8, 16, 32, 64, 128],
            [0.0, 0.25, 0.25, 1.5, 2.5, 2.5, 2.75, 4.75],
            [-0.75, 0.75, 1.5, 2.25, 3.25, 4.5, 5.75, 6.0],
            [925.0, 929.5, 934.25, 940.0, 941.25, 941.25, 944.75, 946.25],
            [5.5, 2.75, 1.25, 0.0, -0.25, -0.5, -1.75, -2.0],
            [23.0, 23.5, 29.5, 31.25, 34.75, 42.0, 48.0, 52.25],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [-0.25, 1.75, 2.25, 3.5, 4.75, 5.5, 6.75, 7.25]
            ]
        assert len(data) == 8
        assert all(len(d) == 8 for d in data)

        # If num_points is None, we randomly select a mix of data lengths.
        # But not at random -- we need to consider the stats.order functions
        # which take different paths depending on whether their data argument
        # has length 0, 1, 2, or 3 modulo 4. (Or 0, 1 modulo 2.) We make sure
        # we cover each of those cases.
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

    def testNoArgs(self):
        # Fail if given no arguments.
        self.assertRaises(TypeError, self.func)

    @handle_extra_arguments
    def testEmptyData(self, *args):
        # Fail when the first argument is empty.
        for empty in ([], (), iter([])):
            self.assertRaises(ValueError, self.func, empty, *args)

    @handle_extra_arguments
    def testSingleData(self, *args):
        # Pass when the first argument is a single data point.
        for data in self.make_data(1):
            assert len(data) == 1
            _ = self.func(list(data), *args)

    @handle_extra_arguments
    def testDoubleData(self, *args):
        # Pass when the first argument is two data points.
        for x,y in self.make_data(2):
            _ = self.func([x,y], *args)

    @handle_extra_arguments
    def testTripleData(self, *args):
        # Pass when the first argument is three data points.
        for x,y,z in self.make_data(3):
            _ = self.func([x, y, z], *args)

    @handle_data_sets(None)
    @handle_extra_arguments
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
    @handle_extra_arguments
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
    @handle_extra_arguments
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
    @handle_extra_arguments
    def testNumericTypeDoesntMatter(self, data, *args):
        # Test that the type of numeric data shouldn't effect the result.
        expected = self.func(data, *args)
        class MyFloat(float):
            pass
        data = [MyFloat(x) for x in data]
        result = self.func(data, *args)
        # self.assertApproxEqual(data, saved_data, tol=1e-13, rel=None)
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


class SingleDataFailMixin:
    # Test that the test function fails with a single data point.
    # This class overrides the method with the same name in
    # UnivariateMixin.

    @handle_extra_arguments
    def testSingleData(self, *args):
        # Fail when given a single data point.
        for x in (1.0, 0.0, -2.5, 5.5):
            self.assertRaises(ValueError, self.func, [x], *args)


class DoubleDataFailMixin(SingleDataFailMixin):
    # Test that the test function fails with one or two data points.
    # This class overrides the methods with the same names in
    # UnivariateMixin.

    @handle_extra_arguments
    def testDoubleData(self, *args):
        # Fail when the first argument is two data points.
        for x, y in ((1.0, 0.0), (-2.5, 5.5), (2.3, 4.2)):
            self.assertRaises(ValueError, self.func, [x, y], *args)

