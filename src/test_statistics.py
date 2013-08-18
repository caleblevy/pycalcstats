"""Test suite for statistics module."""

#from test.support import run_unittest

import decimal
import doctest
import math
import random
import types
import unittest

from decimal import Decimal
from fractions import Fraction

# Test helper.
from test_statistics_approx import NumericTestCase

# Module to be tested.
import statistics


# === Tests for the statistics module ===

class GlobalsTest(unittest.TestCase):
    module = statistics
    expected_metadata = [
            "__doc__", "__all__", "__version__", "__date__",
            "__author__", "__author_email__",
            ]

    def testMeta(self):
        # Test for the existence of metadata.
        for meta in self.expected_metadata:
            self.assertTrue(hasattr(self.module, meta),
                            "%s not present" % meta)

    def testCheckAll(self):
        # Check everything in __all__ exists and is public.
        module = self.module
        for name in module.__all__:
            # No private names in __all__:
            self.assertFalse(name.startswith("_"),
                             'private name "%s" in __all__' % name)
            # And anything in __all__ must exist:
            self.assertTrue(hasattr(module, name),
                            'missing name "%s" in __all__' % name)


class DocTests(unittest.TestCase):
    def testDocTests(self):
        failed, tried = doctest.testmod(statistics)
        self.assertTrue(tried > 0)
        self.assertTrue(failed == 0)


class StatsErrorTest(unittest.TestCase):
    def testHasException(self):
        self.assertTrue(hasattr(statistics, 'StatisticsError'))
        self.assertTrue(issubclass(statistics.StatisticsError, ValueError))


# === Tests for private utility functions ===

class AddPartialTest(unittest.TestCase):
    def testInplace(self):
        # Test that add_partial modifies list in place and returns None.
        L = []
        result = statistics.add_partial(1.5, L)
        self.assertEqual(L, [0.0, 1.5])
        self.assertTrue(result is None)

    def testAdd(self):
        # Test that add_partial actually does add.
        L = []
        statistics.add_partial(1.5, L)
        statistics.add_partial(2.5, L)
        self.assertEqual(sum(L), 4.0)
        statistics.add_partial(1e120, L)
        statistics.add_partial(1e-120, L)
        statistics.add_partial(0.5, L)
        self.assertEqual(sum(L), 1e120)
        statistics.add_partial(-1e120, L)
        self.assertEqual(sum(L), 4.5)
        statistics.add_partial(-4.5, L)
        self.assertEqual(sum(L), 1e-120)

    def testNan(self):
        # Test that add_partial works as expected with NANs.
        L = []
        for x in (1.5, float('NAN'), 2.5):
            statistics.add_partial(x, L)
        self.assertTrue(math.isnan(sum(L)))

    def do_inf_test(self, infinity):
        L = []
        statistics.add_partial(1.5, L)
        statistics.add_partial(infinity, L)
        statistics.add_partial(2.5, L)
        total = sum(L)
        # Result is an infinity of the correct sign.
        self.assertTrue(math.isinf(total))
        self.assertTrue((total > 0) == (infinity > 0))
        # Adding another infinity doesn't change that.
        statistics.add_partial(infinity, L)
        total = sum(L)
        self.assertTrue(math.isinf(total))
        self.assertTrue((total > 0) == (infinity > 0))
        # But adding an infinity of the opposite sign changes it to a NAN.
        statistics.add_partial(-infinity, L)
        self.assertTrue(math.isnan(sum(L)))

    def testInf(self):
        # Test that add_partial works as expected with INFs.
        inf = float('inf')
        self.do_inf_test(inf)
        self.do_inf_test(-inf)


# === Tests for public functions ===

class UnivariateCommonMixin:
    # Common tests for most univariate functions that take a data argument.

    def testNoArgs(self):
        # Fail if given no arguments.
        self.assertRaises(TypeError, self.func)

    def testEmptyData(self):
        # Fail when the data argument (first argument) is empty.
        for empty in ([], (), iter([])):
            self.assertRaises(statistics.StatisticsError, self.func, empty)

    def prepare_data(self):
        """Return int data for various tests."""
        data = list(range(10))
        while data == sorted(data):
            random.shuffle(data)
        return data

    def testNoInPlaceModifications(self):
        # Test that the function does not modify its input data.
        data = self.prepare_data()
        assert len(data) != 1  # Necessary to avoid infinite loop.
        assert data != sorted(data)
        saved = data[:]
        assert data is not saved
        _ = self.func(data)
        self.assertListEqual(data, saved, "data has been modified")

    def testOrderDoesntMatter(self):
        # Test that the order of data points doesn't change the result.

        # CAUTION: due to floating point rounding errors, the result actually
        # may depend on the order. Consider this test representing an ideal.
        # To avoid this test failing, only test with exact values such as ints
        # or Fractions.
        data = [1, 2, 3, 3, 3, 4, 5, 6]*100
        expected = self.func(data)
        random.shuffle(data)
        actual = self.func(data)
        self.assertEqual(expected, actual)

    def testTypeOfDataCollection(self):
        # Test that the type of iterable data doesn't effect the result.
        class MyList(list):
            pass
        class MyTuple(tuple):
            pass
        def generator(data):
            return (obj for obj in data)
        data = self.prepare_data()
        expected = self.func(data)
        for kind in (list, tuple, iter, MyList, MyTuple, generator):
            result = self.func(kind(data))
            self.assertEqual(result, expected)

    def testRangeData(self):
        # Test that functions work with range objects.
        data = range(20, 50, 3)
        expected = self.func(list(data))
        self.assertEqual(self.func(data), expected)

    def testBadArgTypes(self):
        # Test that function raises when given data of the wrong type.

        # Don't roll the following into a loop like this:
        #   for bad in list_of_bad:
        #       self.check_for_type_error(bad)
        #
        # Since assertRaises doesn't show the arguments that caused the test
        # failure, it is very difficult to debug these test failures when the
        # following are in a loop.
        self.check_for_type_error(None)
        self.check_for_type_error(23)
        self.check_for_type_error(42.0)
        self.check_for_type_error(object())

    def check_for_type_error(self, *args):
        self.assertRaises(TypeError, self.func, *args)

    def testTypeOfDataElement(self):
        # Check the type of data elements doesn't affect the numeric result.
        # This is a weaker test than UnivariateTypeMixin.testTypesConserved,
        # because it checks the numeric result by equality, but not by type.
        class MyFloat(float):
            def __truediv__(self, other):
                return type(self)(super().__truediv__(other))
            def __add__(self, other):
                return type(self)(super().__add__(other))
            __radd__ = __add__

        raw = self.prepare_data()
        expected = self.func(raw)
        for kind in (float, MyFloat, Decimal, Fraction):
            data = [kind(x) for x in raw]
            result = type(expected)(self.func(data))
            self.assertEqual(result, expected)


class UnivariateTypeMixin:
    """Mixin class for type-conserving functions.

    This mixin class holds test(s) for functions which conserve the type of
    individual data points. E.g. the mean of a list of Fractions should itself
    be a Fraction.

    Not all tests to do with types need go in this class. Only those that
    rely on the function returning the same type as its input data.
    """
    def testTypesConserved(self):
        # Test that functions keeps the same type as their data points.
        # (Excludes mixed data types.) This only tests the type of the return
        # result, not the value.
        class MyFloat(float):
            def __truediv__(self, other):
                return type(self)(super().__truediv__(other))
            def __sub__(self, other):
                return type(self)(super().__sub__(other))
            def __rsub__(self, other):
                return type(self)(super().__rsub__(other))
            def __pow__(self, other):
                return type(self)(super().__pow__(other))
            def __add__(self, other):
                return type(self)(super().__add__(other))
            __radd__ = __add__

        data = self.prepare_data()
        for kind in (float, Decimal, Fraction, MyFloat):
            d = [kind(x) for x in data]
            result = self.func(d)
            self.assertIs(type(result), kind)


class TestSum(NumericTestCase, UnivariateCommonMixin, UnivariateTypeMixin):
    # Test cases for statistics.sum() function.

    def setUp(self):
        self.func = statistics.sum

    def testEmptyData(self):
        # Override test for empty data.
        for data in ([], (), iter([])):
            self.assertEqual(self.func(data), 0)
            self.assertEqual(self.func(data, 23), 23)
            self.assertEqual(self.func(data, 2.3), 2.3)

    def testInts(self):
        self.assertEqual(self.func([1, 5, 3, -4, -8, 20, 42, 1]), 60)
        self.assertEqual(self.func([4, 2, 3, -8, 7], 1000), 1008)

    def testFloats(self):
        self.assertEqual(self.func([0.25]*20), 5.0)
        self.assertEqual(self.func([0.125, 0.25, 0.5, 0.75], 1.5), 3.125)

    def testFractions(self):
        F = Fraction
        self.assertEqual(self.func([Fraction(1, 1000)]*500), Fraction(1, 2))

    def testDecimals(self):
        D = Decimal
        data = [D("0.001"), D("5.246"), D("1.702"), D("-0.025"),
                D("3.974"), D("2.328"), D("4.617"), D("2.843"),
                ]
        self.assertEqual(self.func(data), Decimal("20.686"))

    def testDecimalContext(self):
        # Test that sum honours the context settings.
        data = list(map(Decimal, "0.033 0.133 0.233 0.333 0.433".split()))
        with decimal.localcontext(
                decimal.Context(prec=1, rounding=decimal.ROUND_DOWN)
                ):
            self.assertEqual(self.func(data), Decimal("1"))
        with decimal.localcontext(
                decimal.Context(prec=2, rounding=decimal.ROUND_UP)
                ):
            self.assertEqual(self.func(data), Decimal("1.2"))

    def testFloatSum(self):
        # Compare with the math.fsum function.
        # Ideally we ought to get the exact same result, but sometimes
        # we differ by a very slight amount :-(
        data = [random.uniform(-100, 1000) for _ in range(1000)]
        self.assertApproxEqual(self.func(data), math.fsum(data), rel=2e-16)

    def testStartArgument(self):
        # Test that the optional start argument works correctly.
        data = [random.uniform(1, 1000) for _ in range(100)]
        t = self.func(data)
        self.assertEqual(t+42, self.func(data, 42))
        self.assertEqual(t-23, self.func(data, -23))
        self.assertEqual(t+1e20, self.func(data, 1e20))

    def testStringsFail(self):
        # Sum of strings should fail.
        self.assertRaises(TypeError, self.func, [1, 2, 3], '999')
        self.assertRaises(TypeError, self.func, [1, 2, 3, '999'])

    def testBytesFail(self):
        # Sum of bytes should fail.
        self.assertRaises(TypeError, self.func, [1, 2, 3], b'999')
        self.assertRaises(TypeError, self.func, [1, 2, 3, b'999'])

    def testMixedSum(self):
        # Mixed sums are allowed.

        # Careful here: order matters. Can't mix Fraction and Decimal directly,
        # only after they're converted to float.
        data = [1, 2, Fraction(1, 2), 3.0, Decimal("0.25")]
        self.assertEqual(self.func(data), 6.75)


class SumInternalsTest(NumericTestCase):
    # Test internals of the sum function.

    def testIgnoreInstanceFloatMethod(self):
        # Test that __float__ methods on data instances are ignored.

        # Python typically calls __dunder__ methods on the class, not the
        # instance. The ``sum`` implementation calls __float__ directly. To
        # better match the behaviour of Python, we call it only on the class,
        # not the instance. This test will fail if somebody "fixes" that code.

        # Create a fake __float__ method.
        def __float__(self):
            raise AssertionError('test fails')

        # Inject it into an instance.
        class MyNumber(Fraction):
            pass
        x = MyNumber(3)
        x.__float__ = types.MethodType(__float__, x)

        # Check it works as expected.
        self.assertRaises(AssertionError, x.__float__)
        self.assertEqual(float(x), 3.0)
        # And now test the function.
        self.assertEqual(statistics.sum([1.0, 2.0, x, 4.0]), 10.0)


class SumTortureTest(NumericTestCase):
    def testTorture(self):
        # Tim Peters' torture test for sum, and variants of same.
        self.assertEqual(statistics.sum([1, 1e100, 1, -1e100]*10000), 20000.0)
        self.assertEqual(statistics.sum([1e100, 1, 1, -1e100]*10000), 20000.0)
        self.assertApproxEqual(
            statistics.sum([1e-100, 1, 1e-100, -1]*10000), 2.0e-96, rel=5e-16
            )


class SumSpecialValues(NumericTestCase):
    # Test that sum works correctly with IEEE-754 special values.

    def testNan(self):
        for type_ in (float, Decimal):
            nan = type_('nan')
            result = statistics.sum([1, nan, 2])
            self.assertIs(type(result), type_)
            self.assertTrue(math.isnan(result))

    def check_infinity(self, x, inf):
        """Check x is an infinity of the same type and sign as inf."""
        self.assertTrue(math.isinf(x))
        self.assertIs(type(x), type(inf))
        self.assertEqual(x > 0, inf > 0)
        assert x == inf

    def do_test_inf(self, inf):
        # Adding a single infinity gives infinity.
        result = statistics.sum([1, 2, inf, 3])
        self.check_infinity(result, inf)
        # Adding two infinities of the same sign also gives infinity.
        result = statistics.sum([1, 2, inf, 3, inf, 4])
        self.check_infinity(result, inf)

    def testFloatInf(self):
        inf = float('inf')
        for sign in (+1, -1):
            self.do_test_inf(sign*inf)

    def testDecimalInf(self):
        inf = Decimal('inf')
        for sign in (+1, -1):
            self.do_test_inf(sign*inf)

    def testFloatMismatchedInf(self):
        # Test that adding two infinities of opposite sign gives a NAN.
        inf = float('inf')
        result = statistics.sum([1, 2, inf, 3, -inf, 4])
        self.assertTrue(math.isnan(result))

    def testDecimalMismatchedInf(self):
        # Test behaviour of Decimal INFs with opposite sign.
        inf = Decimal('inf')
        data = [1, 2, inf, 3, -inf, 4]
        sum = statistics.sum
        with decimal.localcontext(decimal.ExtendedContext):
            self.assertTrue(math.isnan(sum(data)))
        with decimal.localcontext(decimal.BasicContext):
            self.assertRaises(decimal.InvalidOperation, sum, data)


# === Tests for averages ===

class AverageMixin(UnivariateCommonMixin):
    # Mixin class holding common tests for averages.

    def testSingleValue(self):
        # Average of a single value is the value itself.
        for x in (23, 42.5, 1.3e15, Fraction(15, 19), Decimal('0.28')):
            self.assertEqual(self.func([x]), x)

    def testRepeatedSingleValue(self):
        # The average of a single repeated value is the value itself.
        for x in (3.5, 17, 2.5e15, Fraction(61, 67), Decimal('4.9712')):
            for count in (2, 5, 10, 20):
                data = [x]*count
                self.assertEqual(self.func(data), x)


class TestMean(NumericTestCase, AverageMixin, UnivariateTypeMixin):
    def setUp(self):
        self.func = statistics.mean

    def testTorturePep(self):
        # "Torture Test" from PEP-450.
        self.assertEqual(self.func([1e100, 1, 3, -1e100]), 1)

    def testInts(self):
        # Test mean with ints.
        data = [0, 1, 2, 3, 3, 3, 4, 5, 5, 6, 7, 7, 7, 7, 8, 9]
        random.shuffle(data)
        self.assertEqual(self.func(data), 4.8125)

    def testFloats(self):
        # Test mean with floats.
        data = [17.25, 19.75, 20.0, 21.5, 21.75, 23.25, 25.125, 27.5]
        random.shuffle(data)
        self.assertEqual(self.func(data), 22.015625)

    def testDecimals(self):
        # Test mean with ints.
        D = Decimal
        data = [D("1.634"), D("2.517"), D("3.912"), D("4.072"), D("5.813")]
        random.shuffle(data)
        self.assertEqual(self.func(data), D("3.5896"))

    def testFractions(self):
        # Test mean with Fractions.
        F = Fraction
        data = [F(1, 2), F(2, 3), F(3, 4), F(4, 5), F(5, 6), F(6, 7), F(7, 8)]
        random.shuffle(data)
        self.assertEqual(self.func(data), F(1479, 1960))

    def testInf(self):
        # Test mean with infinities.
        raw = [1, 3, 5, 7, 9]  # Use only ints, to avoid TypeError later.
        for kind in (float, Decimal):
            for sign in (1, -1):
                inf = kind("inf")*sign
                data = raw + [inf]
                result = self.func(data)
                self.assertTrue(math.isinf(result))
                self.assertEqual(result, inf)

    def testMismatchedInfs(self):
        # Test mean with infinities of opposite sign.
        data = [2, 4, 6, float('inf'), 1, 3, 5, float('-inf')]
        result = self.func(data)
        self.assertTrue(math.isnan(result))

    def testNan(self):
        # Test mean with NANs.
        raw = [1, 3, 5, 7, 9]  # Use only ints, to avoid TypeError later.
        for kind in (float, Decimal):
            inf = kind("nan")
            data = raw + [inf]
            result = self.func(data)
            self.assertTrue(math.isnan(result))

    def testBigData(self):
        # Test adding a large constant to every data point.
        c = 1e9
        data = [3.4, 4.5, 4.9, 6.7, 6.8, 7.2, 8.0, 8.1, 9.4]
        expected = self.func(data) + c
        assert expected != c
        result = self.func([x+c for x in data])
        self.assertEqual(result, expected)

    def testDoubledData(self):
        # Mean of [a,b,c...z] should be same as for [a,a,b,b,c,c...z,z].
        data = [random.uniform(-3, 5) for _ in range(1000)]
        expected = self.func(data)
        actual = self.func(data*2)
        self.assertApproxEqual(actual, expected)


class TestMedian(NumericTestCase, AverageMixin):
    # Common tests for median and all median.* functions.
    def setUp(self):
        self.func = statistics.median

    def prepare_data(self):
        """Overload method from UnivariateCommonMixin."""
        data = super().prepare_data()
        if len(data)%2 != 1:
            data.append(2)
        return data

    def testEvenInts(self):
        # Test median with an even number of int data points.
        data = [1, 2, 3, 4, 5, 6]
        assert len(data)%2 == 0
        self.assertEqual(self.func(data), 3.5)

    def testOddInts(self):
        # Test median with an odd number of int data points.
        data = [1, 2, 3, 4, 5, 6, 9]
        assert len(data)%2 == 1
        self.assertEqual(self.func(data), 4)

    def testOddFractions(self):
        # Test median works with an odd number of Fractions.
        F = Fraction
        data = [F(1, 7), F(2, 7), F(3, 7), F(4, 7), F(5, 7)]
        assert len(data)%2 == 1
        random.shuffle(data)
        self.assertEqual(self.func(data), F(3, 7))

    def testEvenFractions(self):
        # Test median works with an even number of Fractions.
        F = Fraction
        data = [F(1, 7), F(2, 7), F(3, 7), F(4, 7), F(5, 7), F(6, 7)]
        assert len(data)%2 == 0
        random.shuffle(data)
        self.assertEqual(self.func(data), F(1, 2))

    def testOddDecimals(self):
        # Test median works with an odd number of Decimals.
        D = Decimal
        data = [D('2.5'), D('3.1'), D('4.2'), D('5.7'), D('5.8')]
        assert len(data)%2 == 1
        random.shuffle(data)
        self.assertEqual(self.func(data), D('4.2'))

    def testEvenDecimals(self):
        # Test median works with an even number of Decimals.
        D = Decimal
        data = [D('1.2'), D('2.5'), D('3.1'), D('4.2'), D('5.7'), D('5.8')]
        assert len(data)%2 == 0
        random.shuffle(data)
        self.assertEqual(self.func(data), D('3.65'))


class TestMedianDataType(NumericTestCase, UnivariateTypeMixin):
    # Test conservation of data element type for median.
    def setUp(self):
        self.func = statistics.median

    def prepare_data(self):
        data = list(range(15))
        assert len(data)%2 == 1
        while data == sorted(data):
            random.shuffle(data)
        return data


class TestMedianLow(TestMedian, UnivariateTypeMixin):
    def setUp(self):
        self.func = statistics.median.low

    def testEvenInts(self):
        # Test median.low with an even number of ints.
        data = [1, 2, 3, 4, 5, 6]
        assert len(data)%2 == 0
        self.assertEqual(self.func(data), 3)

    def testEvenFractions(self):
        # Test median.low works with an even number of Fractions.
        F = Fraction
        data = [F(1, 7), F(2, 7), F(3, 7), F(4, 7), F(5, 7), F(6, 7)]
        assert len(data)%2 == 0
        random.shuffle(data)
        self.assertEqual(self.func(data), F(3, 7))

    def testEvenDecimals(self):
        # Test median.low works with an even number of Decimals.
        D = Decimal
        data = [D('1.1'), D('2.2'), D('3.3'), D('4.4'), D('5.5'), D('6.6')]
        assert len(data)%2 == 0
        random.shuffle(data)
        self.assertEqual(self.func(data), D('3.3'))


class TestMedianHigh(TestMedian, UnivariateTypeMixin):
    def setUp(self):
        self.func = statistics.median.high

    def testEvenInts(self):
        # Test median.high with an even number of ints.
        data = [1, 2, 3, 4, 5, 6]
        assert len(data)%2 == 0
        self.assertEqual(self.func(data), 4)

    def testEvenFractions(self):
        # Test median.high works with an even number of Fractions.
        F = Fraction
        data = [F(1, 7), F(2, 7), F(3, 7), F(4, 7), F(5, 7), F(6, 7)]
        assert len(data)%2 == 0
        random.shuffle(data)
        self.assertEqual(self.func(data), F(4, 7))

    def testEvenDecimals(self):
        # Test median.high works with an even number of Decimals.
        D = Decimal
        data = [D('1.1'), D('2.2'), D('3.3'), D('4.4'), D('5.5'), D('6.6')]
        assert len(data)%2 == 0
        random.shuffle(data)
        self.assertEqual(self.func(data), D('4.4'))


class TestMedianGrouped(TestMedian):
    # Test median.grouped.
    # Doesn't conserve data element types, so don't use TestMedianType.
    def setUp(self):
        self.func = statistics.median.grouped

    def testOddNumberRepeated(self):
        # Test median.grouped with repeated median values.
        data = [12, 13, 14, 14, 14, 15, 15]
        assert len(data)%2 == 1
        self.assertEqual(self.func(data), 14)
        #---
        data = [12, 13, 14, 14, 14, 14, 15]
        assert len(data)%2 == 1
        self.assertEqual(self.func(data), 13.875)
        #---
        data = [5, 10, 10, 15, 20, 20, 20, 20, 25, 25, 30]
        assert len(data)%2 == 1
        self.assertEqual(self.func(data, 5), 19.375)
        #---
        data = [16, 18, 18, 18, 18, 20, 20, 20, 22, 22, 22, 24, 24, 26, 28]
        assert len(data)%2 == 1
        self.assertApproxEqual(self.func(data, 2), 20.66666667, tol=1e-8)

    def testEvenNumberRepeated(self):
        # Test median.grouped with repeated median values.
        data = [5, 10, 10, 15, 20, 20, 20, 25, 25, 30]
        assert len(data)%2 == 0
        self.assertApproxEqual(self.func(data, 5), 19.16666667, tol=1e-8)
        #---
        data = [2, 3, 4, 4, 4, 5]
        assert len(data)%2 == 0
        self.assertApproxEqual(self.func(data), 3.83333333, tol=1e-8)
        #---
        data = [2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6]
        assert len(data)%2 == 0
        self.assertEqual(self.func(data), 4.5)
        #---
        data = [3, 4, 4, 4, 5, 5, 5, 5, 6, 6]
        assert len(data)%2 == 0
        self.assertEqual(self.func(data), 4.75)

    def testRepeatedSingleValue(self):
        # Override method from AverageMixin.
        # Yet again, failure of median.grouped to conserve the data type
        # causes me headaches :-(
        for x in (5.3, 68, 4.3e17, Fraction(29, 101), Decimal('32.9714')):
            for count in (2, 5, 10, 20):
                data = [x]*count
                self.assertEqual(self.func(data), float(x))

    def testOddFractions(self):
        # Test median.grouped works with an odd number of Fractions.
        F = Fraction
        data = [F(5, 4), F(9, 4), F(13, 4), F(13, 4), F(17, 4)]
        assert len(data)%2 == 1
        random.shuffle(data)
        self.assertEqual(self.func(data), 3.0)

    def testEvenFractions(self):
        # Test median.grouped works with an even number of Fractions.
        F = Fraction
        data = [F(5, 4), F(9, 4), F(13, 4), F(13, 4), F(17, 4), F(17, 4)]
        assert len(data)%2 == 0
        random.shuffle(data)
        self.assertEqual(self.func(data), 3.25)

    def testOddDecimals(self):
        # Test median.grouped works with an odd number of Decimals.
        D = Decimal
        data = [D('5.5'), D('6.5'), D('6.5'), D('7.5'), D('8.5')]
        assert len(data)%2 == 1
        random.shuffle(data)
        self.assertEqual(self.func(data), 6.75)

    def testEvenDecimals(self):
        # Test median.grouped works with an even number of Decimals.
        D = Decimal
        data = [D('5.5'), D('5.5'), D('6.5'), D('6.5'), D('7.5'), D('8.5')]
        assert len(data)%2 == 0
        random.shuffle(data)
        self.assertEqual(self.func(data), 6.5)
        #---
        data = [D('5.5'), D('5.5'), D('6.5'), D('7.5'), D('7.5'), D('8.5')]
        assert len(data)%2 == 0
        random.shuffle(data)
        self.assertEqual(self.func(data), 7.0)

    def testInterval(self):
        # Test median.grouped with interval argument.
        data = [2.25, 2.5, 2.5, 2.75, 2.75, 3.0, 3.0, 3.25, 3.5, 3.75]
        self.assertEqual(self.func(data, 0.25), 2.875)
        data = [2.25, 2.5, 2.5, 2.75, 2.75, 2.75, 3.0, 3.0, 3.25, 3.5, 3.75]
        self.assertApproxEqual(self.func(data, 0.25), 2.83333333, tol=1e-8)
        data = [220, 220, 240, 260, 260, 260, 260, 280, 280, 300, 320, 340]
        self.assertEqual(self.func(data, 20), 265.0)


class TestMode(NumericTestCase, AverageMixin, UnivariateTypeMixin):
    # Test cases for the discrete version of mode.
    def setUp(self):
        self.func = statistics.mode

    def prepare_data(self):
        """Overload method from UnivariateCommonMixin."""
        # Make sure test data has exactly one mode.
        return [1, 1, 1, 1, 3, 4, 7, 9, 0, 8, 2]

    def testRangeData(self):
        # Override test from UnivariateCommonMixin.
        data = range(20, 50, 3)
        self.assertRaises(statistics.StatisticsError, self.func, data)
        expected = self.func(list(data), max_modes=0)
        self.assertEqual(self.func(data, max_modes=0), expected)

    def testNominalData(self):
        # Test mode with nominal data.
        data = 'abcbdb'
        self.assertEqual(self.func(data), 'b')
        data = 'fe fi fo fum fi fi'.split()
        self.assertEqual(self.func(data), 'fi')

    def testDiscreteData(self):
        # Test mode with discrete numeric data.
        data = list(range(10))
        for i in range(10):
            d = data + [i]
            random.shuffle(d)
            self.assertEqual(self.func(d), i)

    def testBimodalData(self):
        # Test mode with bimodal data.
        data = [1, 1, 2, 2, 2, 2, 3, 4, 5, 6, 6, 6, 6, 7, 8, 9, 9]
        assert data.count(2) == data.count(6) == 4
        # Check for an exception with the default.
        self.assertRaises(statistics.StatisticsError, self.func, data)
        # Now check for correct results with two modes.
        result = self.func(data, max_modes=2)
        self.assertEqual(sorted(result), [2, 6])

    def testTrimodalData(self):
        # Test mode with trimodal data.
        data = list(range(10))*4 + [1, 5, 8]
        assert data.count(1) == data.count(5) == data.count(8) == 5
        # Check for an exception with max_modes < 3.
        self.assertRaises(statistics.StatisticsError, self.func, data, 1)
        self.assertRaises(statistics.StatisticsError, self.func, data, 2)
        # And check for the correct modes.
        result = self.func(data, max_modes=3)
        self.assertEqual(sorted(result), [1, 5, 8])

    def testUniqueDataFailure(self):
        # Test mode exception when data points are all unique.
        data = list(range(10))
        self.assertRaises(statistics.StatisticsError, self.func, data)

    def testUniqueDataNoFailure(self):
        # Test mode when the data points are all unique.
        data = list(range(10))
        result = self.func(data, max_modes=0)
        self.assertEqual(sorted(data), sorted(result))

    def testNoneData(self):
        # Test that mode raises TypeError if given None as data.

        # This test is necessary because the implementation of mode uses
        # collections.Counter, which accepts None and returns an empty dict.
        self.assertRaises(TypeError, self.func, None)


# === Tests for variances and standard deviations ===

class VarianceStdevMixin(UnivariateCommonMixin):
    # Mixin class holding common tests for variance and std dev.

    # Subclasses should inherit from this before NumericTestClass, in order
    # to see the rel attribute below. See testShiftData for an explanation.

    rel = 1e-12

    def testSingleValue(self):
        # Deviation of a single value is zero.
        for x in (11, 19.8, 4.6e14, Fraction(21, 34), Decimal('8.392')):
            self.assertEqual(self.func([x]), 0)

    def testRepeatedSingleValue(self):
        # The deviation of a single repeated value is zero.
        for x in (7.2, 49, 8.1e15, Fraction(3, 7), Decimal('62.4802')):
            for count in (2, 3, 5, 15):
                data = [x]*count
                self.assertEqual(self.func(data), 0)

    def testDomainErrorRegression(self):
        # Regression test for a domain error exception.
        # (Thanks to Geremy Condra.)
        data = [0.123456789012345]*10000
        # All the items are identical, so variance should be exactly zero.
        # We allow some small round-off error, but not much.
        result = self.func(data)
        self.assertApproxEqual(result, 0.0, tol=5e-17)
        self.assertTrue(result >= 0)  # A negative result must fail.

    def testShiftData(self):
        # Test that shifting the data by a constant amount does not affect
        # the variance or stdev. Or at least not much.

        # Due to rounding, this test should be considered an ideal. We allow
        # some tolerance away from "no change at all" by setting tol and/or rel
        # attributes. Subclasses may set tighter or looser error tolerances.
        raw = [1.03, 1.27, 1.94, 2.04, 2.58, 3.14, 4.75, 4.98, 5.42, 6.78]
        expected = self.func(raw)
        # Don't set shift too high, the bigger it is, the more rounding error.
        shift = 1e5
        data = [x + shift for x in raw]
        self.assertApproxEqual(self.func(data), expected)

    def testShiftDataExact(self):
        # Like testShiftData, but result is always exact.
        raw = [1, 3, 3, 4, 5, 7, 9, 10, 11, 16]
        assert all(x==int(x) for x in raw)
        expected = self.func(raw)
        shift = 10**9
        data = [x + shift for x in raw]
        self.assertEqual(self.func(data), expected)

    def testIterListSame(self):
        # Test that iter data and list data give the same result.

        # This is an explicit test that iterators and lists are treated the
        # same; justification for this test over and above the similar test
        # in UnivariateCommonMixin is that an earlier design had variance and
        # friends swap between one- and two-pass algorithms, which would
        # sometimes give different results.
        data = [random.uniform(-3, 8) for _ in range(1000)]
        expected = self.func(data)
        self.assertEqual(self.func(iter(data)), expected)


class TestPVariance(VarianceStdevMixin, NumericTestCase, UnivariateTypeMixin):
    # Tests for population variance.
    def setUp(self):
        self.func = statistics.pvariance

    def testExactUniform(self):
        # Test the variance against an exact result for uniform data.
        data = list(range(10000))
        random.shuffle(data)
        expected = (10000**2 - 1)/12  # Exact value.
        self.assertEqual(self.func(data), expected)

    def testInts(self):
        # Test population variance with int data.
        data = [4, 7, 13, 16]
        exact = 22.5
        self.assertEqual(self.func(data), exact)

    def testFractions(self):
        # Test population variance with Fraction data.
        F = Fraction
        data = [F(1, 4), F(1, 4), F(3, 4), F(7, 4)]
        exact = F(3, 8)
        result = self.func(data)
        self.assertEqual(result, exact)
        self.assertTrue(isinstance(result, Fraction))

    def testDecimals(self):
        # Test population variance with Decimal data.
        D = Decimal
        data = [D("12.1"), D("12.2"), D("12.5"), D("12.9")]
        exact = D('0.096875')
        result = self.func(data)
        self.assertEqual(result, exact)
        self.assertTrue(isinstance(result, Decimal))


class TestVariance(VarianceStdevMixin, NumericTestCase, UnivariateTypeMixin):
    # Tests for sample variance.
    def setUp(self):
        self.func = statistics.variance

    def testSingleValue(self):
        # Override method from VarianceStdevMixin.
        for x in (35, 24.7, 8.2e15, Fraction(19, 30), Decimal('4.2084')):
            self.assertRaises(statistics.StatisticsError, self.func, [x])

    def testInts(self):
        # Test sample variance with int data.
        data = [4, 7, 13, 16]
        exact = 30
        self.assertEqual(self.func(data), exact)

    def testFractions(self):
        # Test sample variance with Fraction data.
        F = Fraction
        data = [F(1, 4), F(1, 4), F(3, 4), F(7, 4)]
        exact = F(1, 2)
        result = self.func(data)
        self.assertEqual(result, exact)
        self.assertTrue(isinstance(result, Fraction))

    def testDecimals(self):
        # Test sample variance with Decimal data.
        D = Decimal
        data = [D(2), D(2), D(7), D(9)]
        exact = 4*D('9.5')/D(3)
        result = self.func(data)
        self.assertEqual(result, exact)
        self.assertTrue(isinstance(result, Decimal))


class TestPStdev(VarianceStdevMixin, NumericTestCase):
    # Tests for population standard deviation.
    def setUp(self):
        self.func = statistics.pstdev

    def testCompareToVariance(self):
        # Test that stdev is, in fact, the square root of variance.
        data = [random.uniform(-17, 24) for _ in range(1000)]
        expected = math.sqrt(statistics.pvariance(data))
        self.assertEqual(self.func(data), expected)


class TestStdev(VarianceStdevMixin, NumericTestCase):
    # Tests for sample standard deviation.
    def setUp(self):
        self.func = statistics.stdev

    def testSingleValue(self):
        # Override method from VarianceStdevMixin.
        for x in (81, 203.74, 3.9e14, Fraction(5, 21), Decimal('35.719')):
            self.assertRaises(statistics.StatisticsError, self.func, [x])

    def testCompareToVariance(self):
        # Test that stdev is, in fact, the square root of variance.
        data = [random.uniform(-2, 9) for _ in range(1000)]
        expected = math.sqrt(statistics.variance(data))
        self.assertEqual(self.func(data), expected)


# === Run tests ===

def test_main():
    #run_unittest()
    unittest.main()


if __name__ == "__main__":
    test_main()

