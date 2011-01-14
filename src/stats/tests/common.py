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


def handle_data_sets(func):
    # Decorate test methods so that they call the function being tested with
    # various examples of given data. See the comment in the UnivariateMixin
    # test class for more detail.
    test_data = [
        # Make sure we cover the cases len(data)%4 -> 0...3
        range(8),
        range(9),
        range(10),
        range(11),
        # Any extra test data is a bonus.
        [-2, -1, 0, 1, 4, 5, 5, 7, 9],
        [925.0, 929.5, 934.25, 940.0, 941.25, 941.25, 944.75, 946.25],
        [0.25, 0.75, 1.5, 1.5, 2.25, 3.25, 4.5, 5.5, 5.75, 6.0],
        ]
    if __debug__:
        for i in (0, 1, 2, 3):
            assert any(len(x)%4 == i for x in test_data)
    @functools.wraps(func)
    def inner_handle_data_sets(self, *args, **kwargs):
        for data in test_data:
            func(self, list(data), *args, **kwargs)
    return inner_handle_data_sets


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

    # Most functions won't care much about the length of the input data,
    # provided there are sufficient data points (usually >= 1). But when
    # testing the functions in stats.order, we do care about the length:
    # we need to cover all four cases of len(data)%4 = 0, 1, 2, 3.
    # This is handled by the handle_data_sets decorator.

    # This class has the following dependencies:
    #
    #   self.func     - The function being tested, assumed to take at
    #                   least one argument.
    #   self.extras   - (optional) If it exists, a sequence of tuples to
    #                   pass to the test function as extra positional args.
    #
    # If the function needs no extra arguments, just don't define self.extras.
    # Otherwise, each call to the test function will be made 1 or more times,
    # using each tuple taken from self.extras.
    # e.g. if self.extras = [(), (a,), (b,c)] then the function will be
    # called three times per test:
    #   self.func(data)
    #   self.func(data, a)
    #   self.func(data, b, c)
    # (with data set appropriately by the test). This functionality is
    # handled by the handle_extra_arguments decorator.

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
        # Note that this test doesn't care what result is returned, so
        # long as *some* result is returned -- even None.
        for x in (1.0, 0.0, -2.5, 5.5):
            _ = self.func([x], *args)

    @handle_extra_arguments
    def testDoubleData(self, *args):
        # Pass when the first argument is two data points.
        # Note that this test also doesn't care what result is returned.
        for x, y in ((1.0, 0.0), (-2.5, 5.5), (2.3, 4.2)):
            _ = self.func([x, y], *args)

    @handle_extra_arguments
    def testTripleData(self, *args):
        # Pass when the first argument is three data points.
        # Note that this test also doesn't care what result is returned.
        for x,y,z in (
            (1.0, 0.0, -1.5), (-2.5, 5.5, 1.0), (2.3, 4.2, 0.25),
            ):
            _ = self.func([x, y, z], *args)

    @handle_data_sets
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

    @handle_data_sets
    @handle_extra_arguments
    def testOrderDoesntMatter(self, data, *args):
        # Test that the result of the function doesn't depend on the order
        # of data points.
        data.sort()
        expected = self.func(data, *args)
        result = self.func(reversed(data), *args)
        self.assertEqual(expected, result)
        for i in range(10):
            random.shuffle(data)
            result = self.func(data, *args)
            self.assertEqual(expected, result)

    @handle_data_sets
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
            self.assertEqual(expected, result)

    @handle_data_sets
    @handle_extra_arguments
    def testNumericTypeDoesntMatter(self, data, *args):
        # Test that the type of numeric data doesn't effect the result.
        expected = self.func(data, *args)
        class MyFloat(float):
            pass
        data = [MyFloat(x) for x in data]
        result = self.func(data, *args)
        self.assertEqual(expected, result)


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
    # This class overrides the methods with the same name in
    # UnivariateMixin.

    @handle_extra_arguments
    def testDoubleData(self, *args):
        # Fail when the first argument is two data points.
        for x, y in ((1.0, 0.0), (-2.5, 5.5), (2.3, 4.2)):
            self.assertRaises(ValueError, self.func, [x, y], *args)

