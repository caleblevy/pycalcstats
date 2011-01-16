#!/usr/bin/env python3

##  Copyright (c) 2011 Steven D'Aprano.
##  See the file stats/__init__.py for the licence terms for this software.

"""
Test suite for the stats.order module.

"""

import collections
import itertools
import math
import random
import unittest

from stats.tests import NumericTestCase
import stats.tests.common as common

# The module to be tested:
import stats.order


class GlobalsTest(unittest.TestCase, common.GlobalsMixin):
    module = stats.order

    def testDefaultSchemesExist(self):
        # Test for the existence of default scheme globals.
        self.assertTrue(hasattr(self.module, 'QUARTILE_DEFAULT'))
        self.assertTrue(hasattr(self.module, 'QUANTILE_DEFAULT'))



class MedianTest(NumericTestCase, common.UnivariateMixin):
    tol = rel = None  # Default to expect exact equality.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.order.median
        self.data = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
        self.expected = 5.5

    def setUp(self):
        random.shuffle(self.data)

    def testCalculationOdd(self):
        assert len(self.data)%2 == 1
        self.assertEqual(self.func(self.data), self.expected)

    def testCalculationEven(self):
        data = [0.0] + self.data
        assert len(data)%2 == 0
        self.assertEqual(self.func(data), 4.95)

    def testBigData(self):
        data = [x + 1e9 for x in self.data]
        expected = self.expected + 1e9
        assert expected != 1e9  # Avoid catastrophic loss of precision.
        self.assertEqual(self.func(data), expected)

    def testSingleton(self):
        for x in [-1.1, 0.0, 1.1, 2.2, 3.3]:
            self.assertEqual(self.func([x]), x)

    def testDoubling(self):
        # Median of [a,b,c...z] should be same as for [a,a,b,b,c,c...z,z].
        # First try with even number of data points.
        data = [random.random() for _ in range(100)]
        assert len(data)%2 == 0
        a = self.func(data)
        b = self.func(data*2)
        self.assertEqual(a, b)
        # Now with odd number.
        data.append(random.random())
        assert len(data)%2 == 1
        a = self.func(data)
        b = self.func(data*2)
        self.assertEqual(a, b)


class MedianExtrasTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.order.median

    def testMedianOdd(self):
        data = [11, 12, 13, 14, 15, 16, 17, 18, 19]
        assert len(data)%2 == 1
        for x in (-1, 0, 1):
            random.shuffle(data)
            self.assertEqual(self.func(data, x), 15)

    def testMedianLow(self):
        data = [11, 12, 13, 14, 15, 16, 17, 18]
        assert len(data)%2 == 0
        random.shuffle(data)
        self.assertEqual(self.func(data, -1), 14)

    def testMedianNormal(self):
        data = [11, 12, 13, 14, 15, 16, 17, 18]
        assert len(data)%2 == 0
        random.shuffle(data)
        self.assertEqual(self.func(data, 0), 14.5)

    def testMedianHigh(self):
        data = [11, 12, 13, 14, 15, 16, 17, 18]
        assert len(data)%2 == 0
        random.shuffle(data)
        self.assertEqual(self.func(data, 1), 15)


class MidrangeTest(MedianTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.order.midrange

    def testMidrange(self):
        self.assertEqual(self.func([1.0, 2.5]), 1.75)
        self.assertEqual(self.func([1.0, 2.0, 4.0]), 2.5)
        self.assertEqual(self.func([2.0, 4.0, 1.0]), 2.5)
        self.assertEqual(self.func([1.0, 2.5, 3.5, 5.5]), 3.25)
        self.assertEqual(self.func([1.0, 2.5, 3.5, 5.5, 1.5]), 3.25)


class MidhingeTest(common.DoubleDataFailMixin, MedianTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.order.midhinge

    def testMidhinge(self):
        # True hinges occur for n = 4N+5 items, which is 1 modulo 4.
        # We test midhinge on four test data sets, for 1, 2, 3, 0 modulo 4.
        a = [0.1, 0.4, 1.1, 1.4, 2.1, 2.4, 3.1, 3.4, 4.1, 4.4, 5.1, 5.4, 6.1]
        assert len(a) == 4*2 + 5
        b = a + [6.4]
        c = b + [7.1]
        d = c + [7.4]
        for L in (a, b, c, d):
            random.shuffle(L)
        # self.assertApproxEqual(self.func(a), 2.9, tol=1e-10, rel=None)
        self.assertEqual(self.func(a), (1.4+4.4)/2)  # 2.9 plus rounding error.
        self.assertEqual(self.func(b), 3.25)
        self.assertEqual(self.func(c), 3.5)
        self.assertEqual(self.func(d), 3.75)

    def testSingleton(self):
        # Override method in MedianTest.
        self.assertRaises(ValueError, self.func, [23])


class TrimeanTest(
    NumericTestCase, common.DoubleDataFailMixin, common.UnivariateMixin
    ):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.order.trimean

    def generic_sequence_test(self, data, n, expected):
        assert len(data)%4 == n
        random.shuffle(data)
        result = self.func(data)
        self.assertEqual(result, expected)
        data = [x + 1e9 for x in data]
        result = self.func(data)
        self.assertEqual(result, expected+1e9)

    def testSeq0(self):
        data = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8]
        expected = ((2.2+3.3)/2 + 4.4 + 5.5 + (6.6+7.7)/2)/4
        self.generic_sequence_test(data, 0, expected)

    def testSeq1(self):
        data = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
        expected = (3.3 + 5.5*2 + 7.7)/4
        self.generic_sequence_test(data, 1, expected)

    def testSeq2(self):
        data = [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
        expected = (2.2 + 4.4 + 5.5 + 7.7)/4
        self.generic_sequence_test(data, 2, expected)

    def testSeq3(self):
        data = [-1.1, 0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
        expected = ((1.1+2.2)/2 + 4.4*2 + (6.6+7.7)/2)/4
        self.generic_sequence_test(data, 3, expected)

    def testIter(self):
        data = [1.1, 3.3, 4.4, 6.6, 7.7, 9.9]
        expected = (3.3 + 4.4 + 6.6 + 7.7)/4
        self.assertEqual(self.func(iter(data)), expected)


class RangeTest(NumericTestCase, common.UnivariateMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.order.range

    def testSingleton(self):
        for x in (-3.1, 0.0, 4.2, 1.789e12):
            self.assertEqual(self.func([x]), 0)

    def generate_data_sets(self):
        # Yield 2-tuples of (data, expected range).
        # List arguments:
        yield ([42], 0)
        yield ([1, 5], 4)
        yield ([1.5, 4.5, 9.0], 7.5)
        yield ([5, 5, 5], 0)
        data = list(range(500))
        random.shuffle(data)
        for shift in (0, 0.5, 1234.567, -1000, 1e6, 1e9):
            d = [x + shift for x in data]
            yield (d, 499)
        # Subclass of list:
        class MyList(list):
            pass
        yield (MyList([1, 2, 3, 4, 5, 6]), 5)
        yield (MyList([-1, 0, 1, 2, 3, 4, 5]), 6)
        yield (MyList([-1, -2, -3, -4, -5]), 4)
        # Tuple arguments:
        yield ((7, 2), 5)
        yield ((7, 2, 5, 6), 5)
        yield ((3.25, 7.5, 3.25, 4.2), 4.25)
        # range objects:
        yield (range(11), 10)
        yield (range(11, -1, -1), 11)

    def testSequence(self):
        for data, expected in self.generate_data_sets():
            self.assertEqual(self.func(data), expected)

    def testIterator(self):
        for data, expected in self.generate_data_sets():
            self.assertEqual(self.func(iter(data)), expected)


class IQRTest(
    NumericTestCase, common.DoubleDataFailMixin, common.UnivariateMixin
    ):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.order.iqr
        self.extras = [(obj,) for obj in
                       (None, 1, 2, 3, 4, 5, 6, 'excel', 'tukey')]

    def testBadScheme(self):
        # Test that a bad scheme raises an exception.
        for scheme in (-1, 1.5, "spam"):
            self.assertRaises(ValueError, self.func, [1, 2, 3, 4], scheme)
        self.assertRaises(TypeError, self.func, [1, 2, 3, 4], {})

    def testTriplet(self):
        # Test that data consisting of three items behaves as expected.
        data = [1, 5, 12]
        self.assertEqual(self.func(data, 'inclusive'), 5.5)
        self.assertEqual(self.func(data, 'exclusive'), 11)

    def testCaseInsensitive(self):
        data = [1, 2, 3, 6, 9, 12, 18, 22]
        for name, num in stats.order.quartiles.aliases.items():
            a = self.func(data, name.lower())
            b = self.func(data, name.upper())
            c = self.func(data, name.title())
            d = self.func(data, num)
            self.assertEqual(a, b)
            self.assertEqual(a, c)
            self.assertEqual(a, d)

    def testDefaultScheme(self):
        # Test that iqr inherits the global default for quartiles.
        data = list(range(51))
        random.shuffle(data)
        assert len(data) % 4 == 3
        save_scheme = stats.order.QUARTILE_DEFAULT
        schemes = [1, 2, 3, 4, 5, 6, 'inclusive', 'exclusive', 'excel']
        try:
            for scheme in schemes:
                stats.order.QUARTILE_DEFAULT = scheme
                a = self.func(data)
                b = self.func(data, scheme)
                self.assertEqual(a, b)
        finally:
            stats.order.QUARTILE_DEFAULT = save_scheme

    def same_result(self, data, scheme):
        # Check that data gives the same result, no matter what order
        # it is given in.
        assert len(data) > 2
        if len(data) <= 7:
            # Test every permutation exhaustively for small amounts of data.
            perms = itertools.permutations(data)
        else:
            # Take a random sample for larger amounts of data.
            data = list(data)
            perms = []
            for _ in range(50):
                random.shuffle(data)
                perms.append(data[:])
        results = [self.func(perm, scheme) for perm in perms]
        assert len(results) > 1
        self.assertTrue(len(set(results)) == 1)

    def testCompareOrder(self):
        # Ensure results don't depend on the order of the input.
        for scheme in [1, 2, 3, 4, 5, 6]:
            for size in range(3, 12):  # size % 4 -> 3,0,1,2 ...
                self.same_result(range(size), scheme)


class QuartileSkewnessTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.order.quartile_skewness

    def testFailure(self):
        # Test that function raises an exception if the arguments are
        # out of order.
        self.assertRaises(ValueError, self.func, 2, 3, 1)
        self.assertRaises(ValueError, self.func, 9, 8, 7)

    def testNan(self):
        # Test that the degenerate case where all three arguments are
        # equal returns NAN.
        self.assertTrue(math.isnan(self.func(1, 1, 1)))
        self.assertTrue(math.isnan(self.func(5, 5, 5)))

    def testSkew(self):
        # Test skew calculations.
        self.assertEqual(self.func(3, 5, 7), 0.0)
        self.assertEqual(self.func(0, 1, 10), 0.8)
        self.assertEqual(self.func(0, 9, 10), -0.8)


class HingesTest(
    NumericTestCase, common.DoubleDataFailMixin, common.UnivariateMixin
    ):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.order.hinges

    def testQuartileScheme(self):
        # Compare results with those from the quartiles function with the
        # 'hinges' scheme.
        quartiles = stats.order.quartiles
        for n in range(3, 25):
            data = range(n)
            self.assertEqual(self.func(data), quartiles(data, 'hinges'))

    def testHinges(self):
        # Compare results calculated by hand.
        self.assertEqual(self.func([0, 1, 2, 3, 4]), (1, 2, 3))
        self.assertEqual(self.func(range(100, 109)), (102, 104, 106))
        # Now test data that requires interpolation between points.
        self.assertEqual(self.func(range(100, 110)), (102, 104.5, 107))
        self.assertEqual(self.func(range(100, 111)), (102.5, 105, 107.5))
        self.assertEqual(self.func(range(100, 112)), (102.5, 105.5, 108.5))


class DrMathTests(unittest.TestCase):
    # Sample data for testing quartiles taken from Dr Math page:
    # http://mathforum.org/library/drmath/view/60969.html
    # Q2 values are not checked in this test.
    A = range(1, 9)
    B = range(1, 10)
    C = range(1, 11)
    D = range(1, 12)

    def testInclusive(self):
        f = stats.order._Quartiles.inclusive
        q1, _, q3 = f(self.A)
        self.assertEqual(q1, 2.5)
        self.assertEqual(q3, 6.5)
        q1, _, q3 = f(self.B)
        self.assertEqual(q1, 3.0)
        self.assertEqual(q3, 7.0)
        q1, _, q3 = f(self.C)
        self.assertEqual(q1, 3.0)
        self.assertEqual(q3, 8.0)
        q1, _, q3 = f(self.D)
        self.assertEqual(q1, 3.5)
        self.assertEqual(q3, 8.5)

    def testExclusive(self):
        f = stats.order._Quartiles.exclusive
        q1, _, q3 = f(self.A)
        self.assertEqual(q1, 2.5)
        self.assertEqual(q3, 6.5)
        q1, _, q3 = f(self.B)
        self.assertEqual(q1, 2.5)
        self.assertEqual(q3, 7.5)
        q1, _, q3 = f(self.C)
        self.assertEqual(q1, 3.0)
        self.assertEqual(q3, 8.0)
        q1, _, q3 = f(self.D)
        self.assertEqual(q1, 3.0)
        self.assertEqual(q3, 9.0)

    def testMS(self):
        f = stats.order._Quartiles.ms
        q1, _, q3 = f(self.A)
        self.assertEqual(q1, 2)
        self.assertEqual(q3, 7)
        q1, _, q3 = f(self.B)
        self.assertEqual(q1, 3)
        self.assertEqual(q3, 7)
        q1, _, q3 = f(self.C)
        self.assertEqual(q1, 3)
        self.assertEqual(q3, 8)
        q1, _, q3 = f(self.D)
        self.assertEqual(q1, 3)
        self.assertEqual(q3, 9)

    def testMinitab(self):
        f = stats.order._Quartiles.minitab
        q1, _, q3 = f(self.A)
        self.assertEqual(q1, 2.25)
        self.assertEqual(q3, 6.75)
        q1, _, q3 = f(self.B)
        self.assertEqual(q1, 2.5)
        self.assertEqual(q3, 7.5)
        q1, _, q3 = f(self.C)
        self.assertEqual(q1, 2.75)
        self.assertEqual(q3, 8.25)
        q1, _, q3 = f(self.D)
        self.assertEqual(q1, 3.0)
        self.assertEqual(q3, 9.0)

    def testExcel(self):
        f = stats.order._Quartiles.excel
        q1, _, q3 = f(self.A)
        self.assertEqual(q1, 2.75)
        self.assertEqual(q3, 6.25)
        q1, _, q3 = f(self.B)
        self.assertEqual(q1, 3.0)
        self.assertEqual(q3, 7.0)
        q1, _, q3 = f(self.C)
        self.assertEqual(q1, 3.25)
        self.assertEqual(q3, 7.75)
        q1, _, q3 = f(self.D)
        self.assertEqual(q1, 3.5)
        self.assertEqual(q3, 8.5)


class QuartileAliasesTest(unittest.TestCase):
    def testAliasesMapping(self):
        # Test that the quartile function exposes a mapping of aliases.
        self.assertTrue(hasattr(stats.order.quartiles, 'aliases'))
        aliases = stats.order.quartiles.aliases
        self.assertTrue(isinstance(aliases, collections.Mapping))
        self.assertTrue(aliases)

    def testAliasesValues(self):
        allowed_schemes = set(stats.order._Quartiles.QUARTILE_MAP.keys())
        for scheme in stats.order.quartiles.aliases.values():
            self.assertTrue(scheme in allowed_schemes)


class QuartileTest(
    NumericTestCase, common.DoubleDataFailMixin, common.UnivariateMixin
    ):
    # Each test must be run using each scheme, as well as no scheme.
    _schemes = [1, 2, 3, 4, 5, 6, 'excel', 'minitab', None]
    extras  = [(scheme,) for scheme in _schemes]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.order.quartiles

    # Generic tests that don't care about the specific values:

    def testBadScheme(self):
        # Test that invalid schemes will fail.
        data = range(20)
        for scheme in ('', 'spam', 1.5, -1.5, -2):
            self.assertRaises(ValueError, self.func, data, scheme)

    def testCaseInsensitive(self):
        # Test that string scheme aliases are case-insensitive.
        data = range(20)
        for scheme in self.func.aliases:
            a = self.func(data, scheme.lower())
            b = self.func(data, scheme.upper())
            c = self.func(data, scheme.title())
            self.assertEqual(a, b)
            self.assertEqual(a, c)

    def testDefaultScheme(self):
        data = list(range(50))
        random.shuffle(data)
        save_scheme = stats.order.QUARTILE_DEFAULT
        schemes = [1, 2, 3, 4, 5, 6]
        try:
            for scheme in schemes:
                stats.order.QUARTILE_DEFAULT = scheme
                a = self.func(data)
                b = self.func(data, scheme)
                self.assertEqual(a, b)
        finally:
            stats.order.QUARTILE_DEFAULT = save_scheme

    # Tests where we check for the correct result.

    def testInclusive(self):
        # Test the inclusive method of calculating quartiles.
        f = self.func
        scheme = 1
        self.assertEqual(f([0, 1, 2], scheme), (0.5, 1, 1.5))
        self.assertEqual(f([0, 1, 2, 3], scheme), (0.5, 1.5, 2.5))
        self.assertEqual(f([0, 1, 2, 3, 4], scheme), (1, 2, 3))
        self.assertEqual(f([0, 1, 2, 3, 4, 5], scheme), (1, 2.5, 4))
        self.assertEqual(f([0, 1, 2, 3, 4, 5, 6], scheme), (1.5, 3, 4.5))
        self.assertEqual(f(range(1, 9), scheme), (2.5, 4.5, 6.5))
        self.assertEqual(f(range(1, 10), scheme), (3, 5, 7))
        self.assertEqual(f(range(1, 11), scheme), (3, 5.5, 8))
        self.assertEqual(f(range(1, 12), scheme), (3.5, 6, 8.5))
        self.assertEqual(f(range(1, 13), scheme), (3.5, 6.5, 9.5))
        self.assertEqual(f(range(1, 14), scheme), (4, 7, 10))
        self.assertEqual(f(range(1, 15), scheme), (4, 7.5, 11))
        self.assertEqual(f(range(1, 16), scheme), (4.5, 8, 11.5))

    def testExclusive(self):
        # Test the exclusive method of calculating quartiles.
        f = self.func
        scheme = 2
        self.assertEqual(f([0, 1, 2], scheme), (0, 1, 2))
        self.assertEqual(f([0, 1, 2, 3], scheme), (0.5, 1.5, 2.5))
        self.assertEqual(f([0, 1, 2, 3, 4], scheme), (0.5, 2, 3.5))
        self.assertEqual(f([0, 1, 2, 3, 4, 5], scheme), (1, 2.5, 4))
        self.assertEqual(f([0, 1, 2, 3, 4, 5, 6], scheme), (1, 3, 5))
        self.assertEqual(f(range(1, 9), scheme), (2.5, 4.5, 6.5))
        self.assertEqual(f(range(1, 10), scheme), (2.5, 5, 7.5))
        self.assertEqual(f(range(1, 11), scheme), (3, 5.5, 8))
        self.assertEqual(f(range(1, 12), scheme), (3, 6, 9))
        self.assertEqual(f(range(1, 13), scheme), (3.5, 6.5, 9.5))
        self.assertEqual(f(range(1, 14), scheme), (3.5, 7, 10.5))
        self.assertEqual(f(range(1, 15), scheme), (4, 7.5, 11))
        self.assertEqual(f(range(1, 16), scheme), (4, 8, 12))

    def testMS(self):
        f = self.func
        scheme = 3
        self.assertEqual(f(range(3), scheme), (0, 1, 2))
        self.assertEqual(f(range(4), scheme), (0, 1, 3))
        self.assertEqual(f(range(5), scheme), (1, 2, 3))
        self.assertEqual(f(range(6), scheme), (1, 3, 4))
        self.assertEqual(f(range(7), scheme), (1, 3, 5))
        self.assertEqual(f(range(8), scheme), (1, 3, 6))
        self.assertEqual(f(range(9), scheme), (2, 4, 6))
        self.assertEqual(f(range(10), scheme), (2, 5, 7))
        self.assertEqual(f(range(11), scheme), (2, 5, 8))
        self.assertEqual(f(range(12), scheme), (2, 5, 9))

    def testMinitab(self):
        f = self.func
        scheme = 4
        self.assertEqual(f(range(3), scheme), (0, 1, 2))
        self.assertEqual(f(range(4), scheme), (0.25, 1.5, 2.75))
        self.assertEqual(f(range(5), scheme), (0.5, 2, 3.5))
        self.assertEqual(f(range(6), scheme), (0.75, 2.5, 4.25))
        self.assertEqual(f(range(7), scheme), (1, 3, 5))
        self.assertEqual(f(range(8), scheme), (1.25, 3.5, 5.75))
        self.assertEqual(f(range(9), scheme), (1.5, 4, 6.5))
        self.assertEqual(f(range(10), scheme), (1.75, 4.5, 7.25))
        self.assertEqual(f(range(11), scheme), (2, 5, 8))
        self.assertEqual(f(range(12), scheme), (2.25, 5.5, 8.75))

    def testExcel(self):
        f = self.func
        scheme = 5
        # Results generated with OpenOffice.
        self.assertEqual(f(range(3), scheme), (0.5, 1, 1.5))
        self.assertEqual(f(range(4), scheme), (0.75, 1.5, 2.25))
        self.assertEqual(f(range(5), scheme), (1, 2, 3))
        self.assertEqual(f(range(6), scheme), (1.25, 2.5, 3.75))
        self.assertEqual(f(range(7), scheme), (1.5, 3, 4.5))
        self.assertEqual(f(range(8), scheme), (1.75, 3.5, 5.25))
        self.assertEqual(f(range(9), scheme), (2, 4, 6))
        self.assertEqual(f(range(10), scheme), (2.25, 4.5, 6.75))
        self.assertEqual(f(range(11), scheme), (2.5, 5, 7.5))
        self.assertEqual(f(range(12), scheme), (2.75, 5.5, 8.25))
        self.assertEqual(f(range(13), scheme), (3, 6, 9))
        self.assertEqual(f(range(14), scheme), (3.25, 6.5, 9.75))
        self.assertEqual(f(range(15), scheme), (3.5, 7, 10.5))

    def testLangford(self):
        f = self.func
        scheme = 6
        self.assertEqual(f(range(3), scheme), (0, 1, 2))
        self.assertEqual(f(range(4), scheme), (0.5, 1.5, 2.5))
        self.assertEqual(f(range(5), scheme), (1, 2, 3))
        self.assertEqual(f(range(6), scheme), (1, 2.5, 4))
        self.assertEqual(f(range(7), scheme), (1, 3, 5))
        self.assertEqual(f(range(8), scheme), (1.5, 3.5, 5.5))
        self.assertEqual(f(range(9), scheme), (2, 4, 6))
        self.assertEqual(f(range(10), scheme), (2, 4.5, 7))
        self.assertEqual(f(range(11), scheme), (2, 5, 8))
        self.assertEqual(f(range(12), scheme), (2.5, 5.5, 8.5))

    def testBig(self):
        data = list(range(1001, 2001))
        assert len(data) == 1000
        assert len(data)%4 == 0
        random.shuffle(data)
        self.assertEqual(self.func(data, 1), (1250.5, 1500.5, 1750.5))
        self.assertEqual(self.func(data, 2), (1250.5, 1500.5, 1750.5))
        data.append(2001)
        random.shuffle(data)
        self.assertEqual(self.func(data, 1), (1251, 1501, 1751))
        self.assertEqual(self.func(data, 2), (1250.5, 1501, 1751.5))
        data.append(2002)
        random.shuffle(data)
        self.assertEqual(self.func(data, 1), (1251, 1501.5, 1752))
        self.assertEqual(self.func(data, 2), (1251, 1501.5, 1752))
        data.append(2003)
        random.shuffle(data)
        self.assertEqual(self.func(data, 1), (1251.5, 1502, 1752.5))
        self.assertEqual(self.func(data, 2), (1251, 1502, 1753))


class QuantileTest(
    NumericTestCase, common.SingleDataFailMixin, common.UnivariateMixin
    ):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.order.quantile
        self.extras = [(f,) for f in (0.1, 0.2, 0.45, 0.5, 0.55, 0.7, 0.99)]

    def testQuantileArgOutOfRange(self):
        data = [1, 2, 3, 4]
        self.assertRaises(ValueError, self.func, data, -0.1)
        self.assertRaises(ValueError, self.func, data, 1.1)

    def testDefaultScheme(self):
        data = list(range(51))
        random.shuffle(data)
        assert len(data) % 4 == 3
        save_scheme = stats.order.QUANTILE_DEFAULT
        schemes = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            'excel', 'minitab',
            (0.375, 0.25, 0, 1),
            ]
        try:
            for scheme in schemes:
                p = random.random()
                stats.order.QUANTILE_DEFAULT = scheme
                a = self.func(data, p)
                b = self.func(data, p, scheme)
                self.assertEqual(a, b)
        finally:
            stats.order.QUANTILE_DEFAULT = save_scheme

    def testUnsorted(self):
        data = [3, 4, 2, 1, 0, 5]
        assert data != sorted(data)
        self.assertEqual(self.func(data, 0.1, scheme=1), 0)
        self.assertEqual(self.func(data, 0.9, scheme=1), 5)
        self.assertEqual(self.func(data, 0.1, scheme=7), 0.5)
        self.assertEqual(self.func(data, 0.9, scheme=7), 4.5)

    def testIter(self):
        self.assertEqual(self.func(range(12), 0.3, scheme=1), 3)
        self.assertEqual(self.func(range(12), 0.3, scheme=7), 3.3)

    def testUnitInterval(self):
        data = [0, 1]
        for f in (0.01, 0.1, 0.2, 0.25, 0.5, 0.55, 0.8, 0.9, 0.99):
            result = self.func(data, f, scheme=7)
            self.assertApproxEqual(result, f, tol=1e-9, rel=None)

    # For the life of me I can't remember what LQD stands for...
    def testLQD(self):
        expected = [1.0, 1.7, 3.9, 6.1, 8.3, 10.5, 12.7, 14.9, 17.1, 19.3, 20.0]
        ps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        data = range(1, 21)
        for i, p in enumerate(ps):
            result = self.func(data, p, scheme=10)
            self.assertApproxEqual(expected[i], result, tol=1e-12, rel=None)


class QuantilesCompareWithR(NumericTestCase):
    # Compare results of calling quantile() against results from R.
    tol = 1e-3
    rel = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read_data('support/quantiles.dat')
        self.func = stats.order.quantile

    def read_data(self, filename):
        # Read data from external test data file generated using R.
        # First we have to find our location...
        import os
        import stats.tests
        location = os.path.split(stats.tests.__file__)[0]
        # Now add the location to it.
        location = os.path.join(location, filename)
        # Now read the data from that file.
        expected = {}
        with open(location, 'r') as data:
            for line in data:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if '=' in line:
                    label, items = line.split("=")
                    label = label.strip()
                    if label == 'seq':
                        start, end, step = [int(s) for s in items.split()]
                        end += 1
                        self.data = list(range(start, end, step))
                    elif label == 'p':
                        self.fractiles = [float(s) for s in items.split()]
                else:
                    scheme, results = line.split(":")
                    scheme = int(scheme.strip())
                    assert 1 <= scheme <= 9
                    results = [float(x) for x in results.split()]
                    expected[scheme] = results
        self.expected = expected

    def compare(self, scheme):
        fractiles = self.fractiles
        a = [self.func(self.data, p, scheme=scheme) for p in fractiles]
        b = self.expected[scheme]
        for actual, expected in zip(a, b):
            self.assertApproxEqual(actual, expected)

    def testR1(self):  self.compare(1)
    def testR2(self):  self.compare(2)
    def testR3(self):  self.compare(3)
    def testR4(self):  self.compare(4)
    def testR5(self):  self.compare(5)
    def testR6(self):  self.compare(6)
    def testR7(self):  self.compare(7)
    def testR8(self):  self.compare(8)
    def testR9(self):  self.compare(9)


class CompareQuantileMethods(NumericTestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.order.quantile

    def compareMethods(self, scheme, params):
        fractions = [0.0, 0.01, 0.1, 0.2, 0.25, 0.31, 0.42, 0.5,
                     0.55, 0.62, 0.75, 0.83, 0.9, 0.95, 0.99, 1.0]
        data0 = list(range(2000, 2701, 100))
        assert len(data0)%4 == 0
        data1 = list(range(2000, 2801, 100))
        assert len(data1)%4 == 1
        data2 = list(range(2000, 2901, 100))
        assert len(data2)%4 == 2
        data3 = list(range(2000, 3001, 100))
        assert len(data3)%4 == 3
        for data in (data0, data1, data2, data3):
            name = 'data%d' % (len(data) % 4)
            for p in fractions:
                a = self.func(data, p, scheme=scheme)
                b = self.func(data, p, scheme=params)
                self.assertEqual(a, b,
                "Failed for %s with %s != %s; p=%f" % (name, a, b, p))

    def testR1(self):
        scheme = 1; params = (0, 0, 1, 0)
        self.compareMethods(scheme, params)

    # Note that there is no test for R2, as it is not supported by the
    # Mathematica parameterized quantile algorithm.

    @unittest.skip('test currently broken for unknown reasons')
    def testR3(self):
        scheme = 3; params = (0.5, 0, 0, 0)
        self.compareMethods(scheme, params)

    def testR4(self):
        scheme = 4; params = (0, 0, 0, 1)
        self.compareMethods(scheme, params)

    def testR5(self):
        scheme = 5; params = (0.5, 0, 0, 1)
        self.compareMethods(scheme, params)

    def testR6(self):
        scheme = 6; params = (0, 1, 0, 1)
        self.compareMethods(scheme, params)

    def testR7(self):
        scheme = 7; params = (1, -1, 0, 1)
        self.compareMethods(scheme, params)

    def testR8(self):
        scheme = 8; params = (1/3, 1/3, 0, 1)
        self.compareMethods(scheme, params)

    def testR9(self):
        scheme = 9; params = (3/8, 0.25, 0, 1)
        self.compareMethods(scheme, params)


class DecileTest(
    NumericTestCase, common.SingleDataFailMixin, common.UnivariateMixin
    ):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.order.decile
        self.extras = [(n,) for n in range(1, 11)]

    def testExact(self):
        data = range(1, 11)
        assert len(data) == 10
        # No interpolation needed regardless of scheme.
        for i in range(1, 11):
            self.assertEqual(self.func(data, i, scheme=1), i)


class PercentileTest(
    NumericTestCase, common.SingleDataFailMixin, common.UnivariateMixin
    ):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = stats.order.percentile
        self.extras = [(n,) for n in range(3, 101, 7)]

    def testExact(self):
        data = range(1, 101)
        assert len(data) == 100
        # No interpolation needed regardless of scheme.
        for i in range(1, 101):
            self.assertEqual(self.func(data, i, scheme=1), i)

