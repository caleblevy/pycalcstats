#!/usr/bin/env python3

##  Copyright (c) 2011 Steven D'Aprano.
##  See the file stats/__init__.py for the licence terms for this software.

"""
Test suite for the stats.order module.

"""

import random
import unittest

import stats.utils


class MinmaxTest(unittest.TestCase):
    """Tests for minmax function."""
    data = list(range(100))
    expected = (0, 99)

    def key(self, n):
        # Tests assume this is a monotomically increasing function.
        return n*33 - 11

    def setUp(self):
        self.minmax = stats.utils.minmax
        random.shuffle(self.data)

    def testArgsNoKey(self):
        # Test minmax works with multiple arguments and no key.
        self.assertEqual(self.minmax(*self.data), self.expected)

    def testSequenceNoKey(self):
        # Test minmax works with a single sequence argument and no key.
        self.assertEqual(self.minmax(self.data), self.expected)

    def testIterNoKey(self):
        # Test minmax works with a single iterator argument and no key.
        self.assertEqual(self.minmax(iter(self.data)), self.expected)

    def testArgsKey(self):
        # Test minmax works with multiple arguments and a key function.
        result = self.minmax(*self.data, key=self.key)
        self.assertEqual(result, self.expected)

    def testSequenceKey(self):
        # Test minmax works with a single sequence argument and a key.
        result = self.minmax(self.data, key=self.key)
        self.assertEqual(result, self.expected)

    def testIterKey(self):
        # Test minmax works with a single iterator argument and a key.
        it = iter(self.data)
        self.assertEqual(self.minmax(it, key=self.key), self.expected)

    def testCompareNoKey(self):
        # Test minmax directly against min and max built-ins.
        data = random.sample(range(-5000, 5000), 300)
        expected = (min(data), max(data))
        result = self.minmax(data)
        self.assertEqual(result, expected)
        random.shuffle(data)
        result = self.minmax(iter(data))
        self.assertEqual(result, expected)

    def testCompareKey(self):
        # Test minmax directly against min and max built-ins with a key.
        letters = list('abcdefghij')
        random.shuffle(letters)
        assert len(letters) == 10
        data = [count*letter for (count, letter) in enumerate(letters)]
        random.shuffle(data)
        expected = (min(data, key=len), max(data, key=len))
        result = self.minmax(data, key=len)
        self.assertEqual(result, expected)
        random.shuffle(data)
        result = self.minmax(iter(data), key=len)
        self.assertEqual(result, expected)

    def testFailures(self):
        """Test minmax failure modes."""
        self.assertRaises(TypeError, self.minmax)
        self.assertRaises(ValueError, self.minmax, [])
        self.assertRaises(TypeError, self.minmax, 1)


class AddPartialTest(unittest.TestCase):
    def testInplace(self):
        # Test that add_partial modifies list in place and returns None.
        L = []
        result = stats.utils.add_partial(1.5, L)
        self.assertEqual(L, [1.5])
        self.assertTrue(result is None)


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


