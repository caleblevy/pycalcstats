"""Test suite for statistics module."""


import decimal
import doctest
import math
import random
import unittest

from decimal import Decimal
from fractions import Fraction

# Test helper.
from test_approx import NumericTestCase

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

class TestCountIter(unittest.TestCase):

    def get_count(self, it):
        assert isinstance(it, statistics._countiter)
        _ = list(it)
        return it.count

    def testEmpty(self):
        # Test that empty iterables have count of zero.
        it = statistics._countiter([])
        self.assertEqual(self.get_count(it), 0)

    def testNonEmpty(self):
        for n in (1, 2, 8, 2001):
            it = statistics._countiter(range(n))
            self.assertEqual(self.get_count(it), n)


class TestWelford(NumericTestCase):

    def testWelfordMinimal(self):
        # Minimal test case for _welford private function.
        data = [1, 2, 2, 2, 3, 3, 4, 5, 5, 3]
        # NOTE: Don't re-order these values! If you do, the calculation
        #       of ss may change slightly due to rounding.
        # Check the hand-calculated results.
        ss, mean, count = 16, 3, 10
        assert count == len(data)
        assert mean == sum(data)/count
        # Now test that _welford returns the right things.
        results = statistics._welford(statistics._countiter(data))
        self.assertTupleEqual(results, (ss, mean, count))


class TestDirect(NumericTestCase):

    def testDirectMinimal(self):
        # Minimal test case for _direct private function.
        data = [1, 1, 0, 2, 3, 3, 0, 4, 1, 2, 3, 4]
        # Check the hand-calculated results.
        ss, mean, count = 22, 2, 12
        assert count == len(data)
        assert mean == sum(data)/count
        # Now test that _direct returns the right value.
        for n in (None, count):
            result = statistics._direct(data, mean, n)
            self.assertEqual(result, ss)
        

# === Tests for public helper functions ===

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
        statistics.add_partial(1.5, L)
        statistics.add_partial(float('NAN'), L)
        statistics.add_partial(2.5, L)
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


class TestSum(NumericTestCase):
    # Simple test cases for statistics.sum() function.

    def setUp(self):
        self.func = statistics.sum

    def testEmptySum(self):
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

    @unittest.skip('not implemented yet')
    def testDecimals(self):
        pass

    def testTypesConserved(self):
        # Test that sum keeps the same type as its data points.
        data = [1, 3, 2, 0, 7, 5, 4, 3, 8, 9, 1]

        class MyInt(int):
            def __add__(self, other):
                return type(self)(super().__add__(other))
            __radd__ = __add__

        for T in (int, float, Decimal, Fraction, MyInt):
            d = [T(x) for x in data]
            result = self.func(d)
            self.assertIs(type(result), T)

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

class AverageMixin:
    # Mixin class holding common tests for averages.

    def testEmptyData(self):
        # Test that average raises if there is no data.
        self.assertRaises(statistics.StatisticsError, self.func, [])

    def testSingleValue(self):
        # Average of a single value is the value itself.
        for x in (23, 42.5, Fraction(15, 19), Decimal('0.28')):
            self.assertEqual(self.func([x]), x)

    def testTypesConserved(self):
        # Test that averages keep the same type as the arguments.
        data = [1, 3, 2, 0, 7, 5, 4, 3, 8, 9, 1]
        template = 'expected %s but got %s'

        class MyFloat(float):
            def __truediv__(self, other):
                return type(self)(super().__truediv__(other))
            def __add__(self, other):
                return type(self)(super().__add__(other))
            __radd__ = __add__

        for T in (float, Decimal, Fraction, MyFloat):
            d = [T(x) for x in data]
            result = self.func(d)
            self.assertIs(type(result), T, template % (T, type(result)))

    def testOrderDoesntMatter(self):
        # Test that order of data points doesn't change the result.
        # Note: for floats or Decimals, order actually may change the
        #       result if rounding errors are too large. So we only test
        #       this with ints, where we know the results will be exact.
        data = [1, 2, 3, 3, 3, 4, 5, 6]*100
        expected = self.func(data)
        random.shuffle(data)
        actual = self.func(data)
        self.assertEqual(expected, actual)


class TestMean(NumericTestCase, AverageMixin):
    def setUp(self):
        self.func = statistics.mean

    def testTorturePep(self):
        # "Torture Test" from PEP-xxx
        self.assertEqual(self.func([1e100, 1, 3, -1e100]), 1)


class TestMedian(NumericTestCase, AverageMixin):
    def setUp(self):
        self.func = statistics.median

    def testEvenNumber(self):
        # Test median with an even number of data points.
        data = [1, 2, 3, 4, 5, 6]
        assert len(data)%2 == 0
        self.assertEqual(self.func(data), 3.5)

    def testOddNumber(self):
        # Test median with an odd number of data points.
        data = [1, 2, 3, 4, 5, 6, 9]
        assert len(data)%2 == 1
        self.assertEqual(self.func(data), 4)

    def testNoSortInPlace(self):
        # Test that median doesn't sort input list in place.
        data = [6, 7, 8, 9, 1, 2, 3, 4, 5]
        assert data != sorted(data)
        saved = data[:]
        result = self.func(data)
        self.assertListEqual(data, saved)


class TestMedianLow(TestMedian):
    def setUp(self):
        self.func = statistics.median.low

    def testEvenNumber(self):
        # Test median.low with an even number of data points.
        data = [1, 2, 3, 4, 5, 6]
        assert len(data)%2 == 0
        self.assertEqual(self.func(data), 3)


class TestMedianHigh(TestMedian):
    def setUp(self):
        self.func = statistics.median.high

    def testEvenNumber(self):
        # Test median.high with an even number of data points.
        data = [1, 2, 3, 4, 5, 6]
        assert len(data)%2 == 0
        self.assertEqual(self.func(data), 4)


class TestMedianGrouped(TestMedian):
    def setUp(self):
        self.func = statistics.median.grouped

    testTypesConserved = unittest.skip(
            "median.grouped doesn't conserve types"
            )(TestMedian.testTypesConserved)

    def testOddNumberRepeated(self):
        # Test median.grouped with repeated median values.
        data = [12, 13, 14, 14, 14, 15, 15]
        assert len(data)%2 == 1
        self.assertEqual(self.func(data), 14)
        #---
        data = [12, 13, 14, 14, 14, 14, 15]
        assert len(data)%2 == 1
        self.assertEqual(self.func(data), 13.875)

    def testEvenNumberRepeated(self):
        # Test median.grouped with repeated median values.
        data = [2, 3, 4, 4, 4, 5]
        assert len(data)%2 == 0
        self.assertApproxEqual(self.func(data), 3.83333333, tol=1e-8)
        #---
        #data = [2, 3, 4, 4, 4, 5]
        #assert len(data)%2 == 0
        #self.assertApproxEqual(self.func(data), 3.83333333, tol=1e-8)


class TestDiscreteMode(NumericTestCase, AverageMixin):
    # Test cases for the discrete version of mode.
    def setUp(self):
        self.func = statistics.mode

    def testNominalData(self):
        # Test mode with nominal data.
        data = 'abcb'
        self.assertEqual(self.func(data), 'b')
        data = 'fe fi fo fum fi fi'.split()
        self.assertEqual(self.func(data), 'fi')

    def testTypesConserved(self):
        # Test that mode keeps the same type as the arguments.
        data = [1, 2, 2, 2, 3, 4, 3, 2]
        for T in (float, Decimal, Fraction):
            d = [T(x) for x in data]
            result = self.func(d)
            self.assertIs(type(result), T)


# === Run tests ===

def test_main():
    unittest.main()


if __name__ == '__main__':
    test_main()

