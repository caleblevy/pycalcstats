#!/usr/bin/env python3

##  Copyright (c) 2011 Steven D'Aprano.
##  See the file stats/__init__.py for the licence terms for this software.

"""
Common mixin tests for the stats package.

"""


import random


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
    # Common tests for most univariate functions. This should be a
    # conservative set of tests which apply to most functions unless
    # overridden.

    def testNoArgs(self):
        # Fail if given no arguments.
        self.assertRaises(TypeError, self.func)

    def testEmptyData(self):
        # Fail when given a single argument which is empty.
        for empty in ([], (), iter([])):
            self.assertRaises(ValueError, self.func, empty)

    def testSingleData(self):
        # Pass when given a single data point.
        # Note that this test doesn't care what result is returned, so
        # long as *some* result is returned -- even None.
        for x in (1.0, 0.0, -2.5, 5.5):
            _ = self.func([x])

    def testDoubleData(self):
        # Pass when given two data points.
        # Note that this test doesn't care what result is returned, so
        # long as *some* result is returned -- even None.
        for x, y in ((1.0, 0.0), (-2.5, 5.5), (2.3, 4.2)):
            _ = self.func([x, y])

    def testTripleData(self):
        # Pass when given three data points.
        # Note that this test doesn't care what result is returned, so
        # long as *some* result is returned -- even None.
        for x,y,z in ((1.0, 0.0, -1.5), (-2.5, 5.5, 1.0), (2.3, 4.2, 0.25)):
            _ = self.func([x, y, z])

    def testNoInPlaceModifications(self):
        # Test that the function does not modify its input data.
        data = [1.5, 2.5, 0.5, 2.0, 0.5]
        assert data != sorted(data)
        assert data != sorted(data, reverse=True)
        saved_data = data[:]
        assert data is not saved_data
        _ = self.func(data)
        self.assertEqual(data, saved_data)

    def _compareOrderings(self, data):
        # Compare func(data) with various permutations.
        data == sorted(data)
        expected = self.func(data)
        result = self.func(data[::-1])
        self.assertEqual(expected, result)
        for i in range(10):
            random.shuffle(data)
            result = self.func(data)
            self.assertEqual(expected, result)

    def testOrderDoesntMatter(self):
        # Test that the result of the function doesn't depend on the order
        # of data points.
        for data in (
            [-2, 0, 1, 4, 5, 5, 7, 9],
            [0.25, 0.75, 1.5, 1.5, 2.25, 3.25, 4.5, 5.5, 5.75, 6.0],
            [925.0, 929.5, 934.25, 940.0, 941.25, 941.25, 944.75, 946.25],
            ):
            self._compareOrderings(data)

    def _compareTypes(self, data):
        expected = self.func(data)
        class MyList(list):
            pass
        for kind in (list, tuple, iter, MyList):
            result = self.func(kind(data))
            self.assertEqual(expected, result)

    def testTypeDoesntMatter(self):
        # Test that the type of iterable data doesn't effect the result.
        for data in (range(23), range(-35, 36), range(-23, 42, 7)):
            self._compareTypes(data)


class SingleDataFailMixin:
    # Test that the test function fails with a single data point.
    # This class overrides the method with the same name in
    # UnivariateMixin.

    def testSingleData(self):
        # Fail when given a single data point.
        for x in (1.0, 0.0, -2.5, 5.5):
            self.assertRaises(ValueError, self.func, [x])



