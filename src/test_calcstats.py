"""Unit tests for calcstats.py"""

from decimal import Decimal
from fractions import Fraction

import collections
import itertools
import math
import random
import unittest

# Module to be tested:
import calcstats


# A note on coding style
# ----------------------
# Do not use self.fail* unit tests, as they are deprecated in Python 3.2.
# Similarly, avoid plural test cases such as self.testEquals (note the S)
# and friends; although they are not officially deprecated, their use is
# discouraged.


# === Test infrastructure ===

def approx_equal(x, y, tol=1e-12, rel=1e-7):
    """approx_equal(x, y [, tol [, rel]]) => True|False

    Test whether x is approximately equal to y, using an absolute error
    of tol and/or a relative error of rel, whichever is bigger.

    >>> approx_equal(1.2589, 1.2587, 0.003)
    True

    If not given, tol=1e-12 and rel=1e-7.

    Absolute error is defined as abs(x-y); if that is less than or equal to
    tol, x and y are considered approximately equal. If tol is zero, this
    is equivalent to testing x == y.

    Relative error is defined as abs((x-y)/x) or abs((x-y)/y), whichever is
    smaller, provided x or y are not zero. If that figure is less than or
    equal to rel, x and y are considered approximately equal. If rel is zero,
    this is also equivalent to testing x == y.

    (But note that in neither case will x and y be compared directly for
    equality.)

    NANs always compare unequal, even with themselves. Infinities compare
    approximately equal if they have the same sign (both positive or both
    negative). Infinities with different signs compare unequal; so do
    comparisons of infinities with finite numbers.

    tol and rel must be non-negative, finite numbers, otherwise the behaviour
    is undefined.
    """
    # NANs are never equal to anything, approximately or otherwise.
    if math.isnan(x) or math.isnan(y):
        # FIXME Signalling NANs should raise an exception.
        return False
    # Infinities are approximately equal if they have the same sign.
    if math.isinf(x) or math.isinf(y):
        return x == y
    # If we get here, both x and y are finite.
    actual_error = abs(x - y)
    allowed_error = max(tol, rel*max(abs(x), abs(y)))
    return actual_error <= allowed_error


# Generic test suite subclass
# ---------------------------
# We prefer this for testing numeric values that may not be exactly equal.
# Avoid using TestCase.almost_equal, because it sucks :)

class NumericTestCase(unittest.TestCase):
    # By default, we expect exact equality, unless overridden.
    tol = 0
    rel = 0

    def assertApproxEqual(
            self, actual, expected, tol=None, rel=None, msg=None
            ):
        if tol is None: tol = self.tol
        if rel is None: rel = self.rel
        if (
                isinstance(actual, collections.Sequence) and
                isinstance(expected, collections.Sequence)
            ):
            check = self._check_approx_seq
        else:
            check = self._check_approx_num
        check(actual, expected, tol, rel, msg)

    def _check_approx_seq(self, actual, expected, tol, rel, msg):
        if len(actual) != len(expected):
            standardMsg = (
                "actual and expected sequences differ in length;"
                " expected %d items but got %d"
                % (len(expected), len(actual))
                )
            msg = self._formatMessage(msg, standardMsg)
            raise self.failureException(msg)
        for i, (a,e) in enumerate(zip(actual, expected)):
            self._check_approx_num(a, e, tol, rel, msg, i)

    def _check_approx_num(self, actual, expected, tol, rel, msg, idx=None):
        if approx_equal(actual, expected, tol, rel):
            # Test passes. Return early, we are done.
            return None
        # Otherwise we failed.
        standardMsg = self._make_std_err_msg(actual, expected, tol, rel, idx)
        msg = self._formatMessage(msg, standardMsg)
        raise self.failureException(msg)

    @staticmethod
    def _make_std_err_msg(actual, expected, tol, rel, idx):
        # Create the standard error message for approx_equal failures.
        assert actual != expected
        template = (
            'actual value %r differs from expected %r\n'
            '    by more than tol=%r and rel=%r\n'
            '    ..absolute error = %r\n'
            '    ..relative error = %r'
            )
        if idx is not None:
            header = 'numeric sequences first differ at index %d.\n' % idx
            template = header + template
        # Calculate actual errors:
        abs_err = abs(actual - expected)
        base = max(abs(actual), abs(expected))
        if base == 0:
            rel_err = 'inf'
        else:
            rel_err = abs_err/base
        return template % (actual, expected, tol, rel, abs_err, rel_err)


# Here we test the test infrastructure itself.

class ApproxIntegerTest(unittest.TestCase):
    # Test the approx_equal function with ints.

    def _equality_tests(self, x, y):
        """Test ways of spelling 'exactly equal'."""
        return (approx_equal(x, y, tol=0, rel=0),
                approx_equal(y, x, tol=0, rel=0),
                )

    def testExactlyEqual(self):
        # Test that equal values are equal and unequal values are unequal.
        values = [-10**100, -42, -1, 0, 1, 23, 2000, 10**100]
        for x in values:
            results = self._equality_tests(x, x)
            self.assertTrue(all(results), 'equality failure for x=%r' % x)
            results = self._equality_tests(x, x+1)
            self.assertFalse(any(results), 'inequality failure for x=%r' % x)

    def testAbsolute(self):
        # Test approximate equality with an absolute error.
        self.assertTrue(approx_equal(-42, -43, tol=1, rel=0))
        self.assertTrue(approx_equal(15, 16, tol=2, rel=0))
        self.assertFalse(approx_equal(23, 27, tol=3, rel=0))

    def testRelative(self):
        # Test approximate equality with a relative error.
        self.assertTrue(approx_equal(100, 119, tol=0, rel=0.2))
        self.assertTrue(approx_equal(119, 100, tol=0, rel=0.2))
        self.assertFalse(approx_equal(100, 130, tol=0, rel=0.2))
        self.assertFalse(approx_equal(130, 100, tol=0, rel=0.2))

    def testBoth(self):
        # Test approximate equality with both absolute and relative errors.
        a, b = 10.1, 10.15
        # Actual absolute error = 0.05, relative error just under 0.005.
        # (1) compare approx equal with both absolute and relative errors:
        self.assertTrue(approx_equal(a, b, tol=0.1, rel=0.01))
        # (2) compare approx equal with neither absolute nor relative errors:
        self.assertFalse(approx_equal(a, b, tol=0.01, rel=0.001))
        # (3) compare approx equal with absolute but not relative error:
        self.assertTrue(approx_equal(a, b, tol=0.06, rel=0.002))
        # (4) compare approx equal with relative but not absolute error:
        self.assertTrue(approx_equal(a, b, tol=0.04, rel=0.007))

    def testRelSymmetry(self):
        # Check that approx_equal treats relative error symmetrically.
        # (a-b)/a is usually not equal to (a-b)/b. Ensure that this
        # doesn't matter.
        a, b = 23.234, 23.335
        delta = abs(b-a)
        rel_err1, rel_err2 = delta/a, delta/b
        assert rel_err1 > rel_err2
        # Choose an acceptable error margin halfway between the two.
        rel = (rel_err1 + rel_err2)/2
        # Check our logic:
        assert rel*a < delta < rel*b
        # Now see that a and b compare approx equal regardless of which
        # is given first.
        self.assertTrue(approx_equal(a, b, tol=0, rel=rel))
        self.assertTrue(approx_equal(b, a, tol=0, rel=rel))

    def testSymmetry(self):
        # Test that approx_equal(a, b) == approx_equal(b, a)
        alist = [random.random() for _ in range(20)]
        blist = [random.random() for _ in range(20)]
        template = "approx_equal comparisons don't match for %r"
        for a, b in zip(alist, blist):
            for tol in (0, 0.1, 0.7, 1):
                for rel in (0, 0.001, 0.03, 0.4, 1):
                    flag1 = approx_equal(a, b, tol, rel)
                    flag2 = approx_equal(b, a, tol, rel)
                    t = (a, b, tol, rel)
                    self.assertEqual(flag1, flag2, template % (t,))


class ApproxFractionTest(unittest.TestCase):
    # Test the approx_equal function with Fractions.

    def testExactlyEqual(self):
        # Test that equal values are equal and unequal values are unequal.
        F = Fraction
        values = [-F(1, 2), F(0), F(5, 3), F(9, 7), F(35, 36)]
        for x in values:
            self.assertTrue(
                approx_equal(x, x, tol=0, rel=0),
                'equality failure for x=%r' % x
                )
            self.assertFalse(
                approx_equal(x, x+1, tol=0, rel=0),
                'inequality failure for x=%r' % x
                )

    def testAbsolute(self):
        # Test approximate equality with an absolute error.
        F = Fraction
        aeq = approx_equal
        self.assertTrue(aeq(F(7, 9), F(8, 9), tol=F(1, 9), rel=0))
        self.assertTrue(aeq(F(8, 5), F(7, 5), tol=F(2, 5), rel=0))
        self.assertFalse(aeq(F(6, 8), F(8, 8), tol=F(1, 8), rel=0))

    def testRelative(self):
        # Test approximate equality with a relative error.
        F = Fraction
        aeq = approx_equal
        self.assertTrue(aeq(F(45, 100), F(65, 100), tol=0, rel=F(32, 100)))
        self.assertFalse(aeq(F(23, 50), F(48, 50), tol=0, rel=F(26, 50)))


class ApproxDecimalTest(unittest.TestCase):
    # Test the approx_equal function with Decimals.

    def testExactlyEqual(self):
        # Test that equal values are equal and unequal values are unequal.
        D = Decimal
        values = [D('-23.0'), D(0), D('1.3e-15'), D('3.25'), D('1.7e15')]
        for x in values:
            self.assertTrue(
                approx_equal(x, x, tol=0, rel=0),
                'equality failure for x=%r' % x
                )
            self.assertFalse(
                approx_equal(x, x+1, tol=0, rel=0),
                'inequality failure for x=%r' % x
                )

    def testAbsolute(self):
        # Test approximate equality with an absolute error.
        D = Decimal
        aeq = approx_equal
        self.assertTrue(aeq(D('12.78'), D('12.35'), tol=D('0.43'), rel=0))
        self.assertTrue(aeq(D('35.4'), D('36.2'), tol=D('1.5'), rel=0))
        self.assertFalse(aeq(D('35.3'), D('36.2'), tol=D('0.8'), rel=0))

    def testRelative(self):
        # Test approximate equality with a relative error.
        D = Decimal
        aeq = approx_equal
        self.assertTrue(aeq(D('5.4'), D('6.7'), tol=0, rel=D('0.20')))
        self.assertFalse(aeq(D('5.4'), D('6.7'), tol=0, rel=D('0.19')))

    def testSpecials(self):
        nan = Decimal('nan')
        inf = Decimal('inf')
        for y in (nan, inf, -inf, Decimal('1.1')):
            self.assertFalse(approx_equal(nan, y, tol=2, rel=2))
        for y in (nan, -inf, Decimal('1.1')):
            self.assertFalse(approx_equal(inf, y, tol=2, rel=2))
        for y in (nan, inf, Decimal('1.1')):
            self.assertFalse(approx_equal(-inf, y, tol=2, rel=2))
        for y in (nan, inf, -inf):
            self.assertFalse(approx_equal(Decimal('1.1'), y, tol=2, rel=2))
        self.assertTrue(approx_equal(inf, inf, tol=2, rel=2))
        self.assertTrue(approx_equal(-inf, -inf, tol=2, rel=2))


class ApproxFloatTest(unittest.TestCase):
    # Test the approx_equal function with floats.

    def testExactlyEqual(self):
        # Test that equal values are equal and unequal values are unequal.
        values = [-23.0, 0.0, 1.3e-15, 3.37, 1.7e9, 4.7e15]
        for x in values:
            self.assertTrue(
                approx_equal(x, x, tol=0, rel=0),
                'equality failure for x=%r' % x
                )
            self.assertFalse(
                approx_equal(x, x+1, tol=0, rel=0),
                'inequality failure for x=%r' % x
                )

    def testAbsolute(self):
        # Test approximate equality with an absolute error.
        self.assertTrue(approx_equal(4.57, 4.54, tol=0.5, rel=0))
        self.assertTrue(approx_equal(4.57, 4.52, tol=0.5, rel=0))
        self.assertTrue(approx_equal(2.3e12, 2.6e12, tol=0.4e12, rel=0))
        self.assertFalse(approx_equal(2.3e12, 2.6e12, tol=0.2e12, rel=0))
        self.assertTrue(approx_equal(1.01e-9, 1.03e-9, tol=0.05e-9, rel=0))
        self.assertTrue(approx_equal(273.5, 263.9, tol=9.7, rel=0))
        self.assertFalse(approx_equal(273.5, 263.9, tol=9.0, rel=0))

    def testRelative(self):
        # Test approximate equality with a relative error.
        self.assertTrue(approx_equal(3.5, 4.1, tol=0, rel=0.147))
        self.assertFalse(approx_equal(3.5, 4.1, tol=0, rel=0.146))
        self.assertTrue(approx_equal(7.2e11, 6.9e11, tol=0, rel=0.042))
        self.assertFalse(approx_equal(7.2e11, 6.9e11, tol=0, rel=0.041))

    def testSpecials(self):
        nan = float('nan')
        inf = float('inf')
        for y in (nan, inf, -inf, 1.1):
            self.assertFalse(approx_equal(nan, y, tol=2, rel=2))
        for y in (nan, -inf, 1.1):
            self.assertFalse(approx_equal(inf, y, tol=2, rel=2))
        for y in (nan, inf, 1.1):
            self.assertFalse(approx_equal(-inf, y, tol=2, rel=2))
        for y in (nan, inf, -inf):
            self.assertFalse(approx_equal(1.1, y, tol=2, rel=2))
        self.assertTrue(approx_equal(inf, inf, tol=2, rel=2))
        self.assertTrue(approx_equal(-inf, -inf, tol=2, rel=2))

    def testZeroes(self):
        nzero = math.copysign(0.0, -1)
        self.assertTrue(approx_equal(nzero, 0.0, tol=1, rel=1))
        self.assertTrue(approx_equal(0.0, nzero, tol=0, rel=0))


class TestNumericTestCase(unittest.TestCase):
    # The formatting routine that generates the error messages is complex
    # enough that it needs its own test.

        # NOTE: Try not to compare to the exact error message, since
        # that might change. Instead, look for substrings that should
        # be present.

    def test_error_msg(self):
        # Test the error message generated for inexact tests.
        msg = NumericTestCase._make_std_err_msg(2.5, 4.0, 0.5, 0.25, None)
        self.assertIn('actual value 2.5', msg)
        self.assertIn('expected 4.0', msg)
        self.assertIn('tol=0.5', msg)
        self.assertIn('rel=0.25', msg)
        self.assertIn('absolute error = 1.5', msg)
        self.assertIn('relative error = 0.375', msg)

    def test_error_msg_sequence(self):
        # Test the error message generated for sequence tests.
        msg = NumericTestCase._make_std_err_msg(2.5, 4.0, 0.5, 0.25, 7)
        self.assertIn('differ at index 7', msg)
        self.assertIn('actual value 2.5', msg)
        self.assertIn('expected 4.0', msg)
        self.assertIn('tol=0.5', msg)
        self.assertIn('rel=0.25', msg)
        self.assertIn('absolute error = 1.5', msg)
        self.assertIn('relative error = 0.375', msg)

    def testNumericTestCaseIsTestCase(self):
        # Ensure that NumericTestCase actually is a TestCase.
        self.assertTrue(issubclass(NumericTestCase, unittest.TestCase))


# === Utility functions ===

def comp_var(data, p):
    """So-called 'computational formula for variance'.

    FOR TESTING AND COMPARISON USE ONLY, DO NOT USE IN PRODUCTION.

    This formula is numerically unstable and can be extremely inaccurate,
    including returning negative results. Use this only for exact values
    (ints, Fractions) or small data sets with very little rounding error.

    Calculate the population variance σ2 = 1/n**2 * (n*Σ(x**2) - (Σx)**2)

    >>> comp_var([1, 1, 3, 7], 0)
    6.0

    Calculate the sample variance s2 = 1/(n*(n-1)) * (n*Σ(x**2) - (Σx)**2)

    >>> comp_var([1, 1, 3, 7], 1)
    8.0

    """
    n = len(data)
    s1 = sum(x**2 for x in data)
    s2 = sum(data)
    return (n*s1 - s2**2)/(n*(n-p))


class TestCompPVariance(unittest.TestCase):
    """Test the comp_var function.

    Note: any tests here should also be tested against the real variance
    function(s); there's no point in confirming that the computational
    formula doesn't give the right answer if we don't also test that we
    can get the right answer!
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = lambda data: comp_var(data, 0)  # Population variance.
        self.data = [1, 2, 4, 5, 8]
        self.expected = 6.0

    def test_variance(self):
        self.assertEqual(self.func(self.data), self.expected)

    def shifted_data(self):
        return [x+1e12 for x in self.data]*100

    def test_shifted_variance(self):
        # We expect the computational formula to be numerically unstable;
        # if it isn't, we want to know about it!
        data = self.shifted_data()
        variance = self.func(data)
        self.assertTrue(variance < -1e-9)  # Impossible value!


class TestCompVariance(TestCompPVariance):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = lambda data: comp_var(data, 1)  # Sample variance.
        self.expected = 7.5


# === Test metadata, exceptions and module globals ===

class MetadataTest(unittest.TestCase):
    expected_metadata = [
        "__version__", "__date__", "__author__", "__author_email__",
        "__doc__", "__all__",
        ]
    module = calcstats

    def testCheckAll(self):
        # Check everything in __all__ exists.
        module = self.module
        for name in module.__all__:
            # No private names in __all__:
            self.assertFalse(name.startswith("_"),
                             'private name "%s" in __all__' % name)
            # And anything in __all__ must exist:
            self.assertTrue(hasattr(module, name),
                            'missing name "%s" in __all__' % name)

    def testMeta(self):
        # Test for the existence of metadata.
        module = self.module
        for meta in self.expected_metadata:
            self.assertTrue(hasattr(module, meta), "%s not present" % meta)


class StatsErrorTest(unittest.TestCase):
    def testHasException(self):
        self.assertTrue(hasattr(calcstats, 'StatsError'))
        self.assertTrue(issubclass(calcstats.StatsError, ValueError))


# === Test the utility functions ===

class CoroutineTest(unittest.TestCase):
    def testDecorator(self):
        @calcstats.coroutine
        def co():
            x = (yield None)
            y = (yield 42)
        f = co()
        self.assertEqual(f.send(1), 42)


class AddPartialTest(unittest.TestCase):
    def testInplace(self):
        # Test that add_partial modifies list in place and returns None.
        L = []
        result = calcstats.add_partial(L, 1.5)
        self.assertEqual(L, [1.5])
        self.assertTrue(result is None)

    def testAddInts(self):
        # Test that add_partial adds ints.
        ap = calcstats.add_partial
        L = []
        ap(L, 1)
        ap(L, 2)
        self.assertEqual(sum(L), 3)
        ap(L, 1000)
        x = sum(L)
        self.assertEqual(x, 1003)
        self.assertTrue(isinstance(x, int))

    def testAddFloats(self):
        # Test that add_partial adds floats.
        ap = calcstats.add_partial
        L = []
        ap(L, 1.5)
        ap(L, 2.5)
        self.assertEqual(sum(L), 4.0)
        ap(L, 1e120)
        ap(L, 1e-120)
        ap(L, 0.5)
        self.assertEqual(sum(L), 1e120)
        ap(L, -1e120)
        self.assertEqual(sum(L), 4.5)
        ap(L, -4.5)
        self.assertEqual(sum(L), 1e-120)

    def testAddFracs(self):
        # Test that add_partial adds Fractions.
        ap = calcstats.add_partial
        L = []
        ap(L, Fraction(1, 4))
        ap(L, Fraction(2, 3))
        self.assertEqual(sum(L), Fraction(11, 12))
        ap(L, Fraction(42, 23))
        x = sum(L)
        self.assertEqual(x, Fraction(757, 276))
        self.assertTrue(isinstance(x, Fraction))

    def testAddDec(self):
        # Test that add_partial adds Decimals.
        ap = calcstats.add_partial
        L = []
        ap(L, Decimal('1.23456'))
        ap(L, Decimal('6.78901'))
        self.assertEqual(sum(L), Decimal('8.02357'))
        ap(L, Decimal('1e200'))
        ap(L, Decimal('1e-200'))
        self.assertEqual(sum(L), Decimal('1e200'))
        ap(L, Decimal('-1e200'))
        self.assertEqual(sum(L), Decimal('8.02357'))
        ap(L, Decimal('-8.02357'))
        x = sum(L)
        self.assertEqual(x, Decimal('1e-200'))
        self.assertTrue(isinstance(x, Decimal))

    def testAddFloatSubclass(self):
        # Test that add_partial adds float subclass.
        class MyFloat(float):
            def __add__(self, other):
                return MyFloat(super().__add__(other))
            __radd__ = __add__
        ap = calcstats.add_partial
        L = []
        ap(L, MyFloat(1.25))
        ap(L, MyFloat(1e-170))
        ap(L, MyFloat(1e200))
        self.assertEqual(sum(L), 1e200)
        ap(L, MyFloat(5e199))
        ap(L, MyFloat(-1.0))
        ap(L, MyFloat(-2e200))
        ap(L, MyFloat(5e199))
        self.assertEqual(sum(L), 0.25)
        ap(L, MyFloat(-0.25))
        x = sum(L)
        self.assertEqual(x, 1e-170)
        self.assertTrue(isinstance(x, MyFloat))


# === Test sums ===

class TestConsumerMixin:
    def testIsConsumer(self):
        # Test that the function is a consumer.
        cr = self.func()
        self.assertTrue(hasattr(cr, 'send'))


class RunningSumTest(unittest.TestCase, TestConsumerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = calcstats.running_sum

    def testSum(self):
        cr = self.func()
        data = [3, 5, 0, -2, 0.5, 2.75]
        expected = [3, 8, 8, 6, 6.5, 9.25]
        assert len(data)==len(expected)
        for x, y in zip(data, expected):
            self.assertEqual(cr.send(x), y)

    def testSumStart(self):
        start = 3.5
        cr = self.func(start)
        data = [2, 5.5, -4, 0, 0.25, 1.25]
        expected = [2, 7.5, 3.5, 3.5, 3.75, 5.0]
        assert len(data)==len(expected)
        for x, y in zip(data, expected):
            self.assertEqual(cr.send(x), start+y)

    def testSumTortureTest(self):
        cr = self.func()
        for i in range(100):
            self.assertEqual(cr.send(1), 2*i+1)
            self.assertEqual(cr.send(1e100), 1e100)
            self.assertEqual(cr.send(1), 1e100)
            self.assertEqual(cr.send(-1e100), 2*i+2)

    def testFractions(self):
        F = Fraction
        data = [F(3, 5), 2, F(1, 4), F(1, 3), F(3, 2)]
        expected = [F(3, 5), F(13, 5), F(57, 20), F(191, 60), F(281, 60)]
        assert len(data)==len(expected)
        start = F(1, 2)
        rs = self.func(start)
        for f, y in zip(data, expected):
            x = rs.send(f)
            self.assertEqual(x, start+y)
            self.assertTrue(isinstance(x, Fraction))

    def testDecimals(self):
        D = Decimal
        data = [D('0.2'), 3, -D('1.3'), D('2.7'), D('3.2')]
        expected = [D('0.2'), D('3.2'), D('1.9'), D('4.6'), D('7.8')]
        assert len(data)==len(expected)
        start = D('1.555')
        rs = self.func(start)
        for d, y in zip(data, expected):
            x = rs.send(d)
            self.assertEqual(x, start+y)
            self.assertTrue(isinstance(x, Decimal))


class UnivariateMixin:
    # Common tests for most univariate functions that take a data argument.
    #
    # This tests the behaviour of functions of the form func(data [,...])
    # without checking the specific value returned. Testing that the return
    # value is actually correct is not the responsibility of this class.

    def testNoArgs(self):
        # Expect no arguments to raise an exception.
        self.assertRaises(TypeError, self.func)

    def testEmptyData(self):
        # Expect no data points to raise an exception.
        for empty in ([], (), iter([])):
            self.assertRaises(ValueError, self.func, empty)

    def testSingleData(self):
        # Pass if a single data point doesn't raise an exception.
        for data in ([1], [3.3], [1e23]):
            assert len(data) == 1
            _ = self.func(data)

    def testDoubleData(self):
        # Pass if two data points doesn't raise an exception.
        for data in ([1, 3], [3.3, 5.5], [1e23, 2e23]):
            assert len(data) == 2
            _ = self.func(data)

    def testTripleData(self):
        # Pass if three data points doesn't raise an exception.
        for data in ([1, 3, 4], [3.3, 5.5, 6.6], [1e23, 2e23, 1e24]):
            assert len(data) == 3
            _ = self.func(data)

    def testInPlaceModification(self):
        # Test that the function does not modify its input data.
        data = [3, 0, 5, 1, 7, 2]
        # We wish to detect functions that modify the data in place by
        # sorting, which we can't do if the data is already sorted.
        assert data != sorted(data)
        saved = data[:]
        assert data is not saved
        _ = self.func(data)
        self.assertEqual(data, saved, "data has been modified")

    def testOrderOfDataPoints(self):
        # Test that the result of the function shouldn't depend on the
        # order of data points. In practice, due to floating point
        # rounding, it may depend slightly.
        data = [1, 2, 2, 3, 4, 7, 9]
        expected = self.func(data)
        result = self.func(data[::-1])
        self.assertApproxEqual(expected, result)
        for i in range(10):
            random.shuffle(data)
            result = self.func(data)
            self.assertApproxEqual(result, expected)

    def get_allowed_kinds(self):
        class MyList(list):
            pass
        class MyTuple(tuple):
            pass
        def generator(data):
            return (obj for obj in data)
        return (list, tuple, iter, MyList, MyTuple, generator)

    def testTypeOfDataCollection(self):
        # Test that the type of iterable data doesn't effect the result.
        data = range(1, 16, 2)
        expected = self.func(data)
        for kind in self.get_allowed_kinds():
            result = self.func(kind(data))
            self.assertEqual(result, expected)

    def testFloatTypes(self):
        # Test that the type of float shouldn't effect the result.
        class MyFloat(float):
            def __add__(self, other):
                return MyFloat(super().__add__(other))
            __radd__ = __add__
            def __mul__(self, other):
                return MyFloat(super().__mul__(other))
            __rmul__ = __mul__
        data = [2.5, 5.5, 0.25, 1.0, 2.25, 7.0, 7.25]
        expected = self.func(data)
        data = [MyFloat(x) for x in data]
        result = self.func(data)
        self.assertEqual(result, expected)

    # FIXME: needs tests for bad argument types.


class SumTest(NumericTestCase, UnivariateMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = calcstats.sum

    def testEmptyData(self):
        # Override UnivariateMixin method.
        for empty in ([], (), iter([])):
            self.assertEqual(self.func(empty), 0)
            for start in (Fraction(23, 42), Decimal('3.456'), 123.456):
                self.assertEqual(self.func(empty, start), start)

    def testCompareWithFSum(self):
        # Compare with the math.fsum function.
        data = [random.uniform(-500, 5000) for _ in range(1000)]
        actual = self.func(data)
        expected = math.fsum(data)
        self.assertApproxEqual(actual, expected, rel=1e-15)

    def testExactSeries(self):
        # Compare with exact formulae for certain sums of integers.
        # sum of 1, 2, 3, ... n = n(n+1)/2
        data = list(range(1, 131))
        random.shuffle(data)
        expected = 130*131/2
        self.assertEqual(self.func(data), expected)
        # sum of squares of 1, 2, 3, ... n = n(n+1)(2n+1)/6
        data = [n**2 for n in range(1, 57)]
        random.shuffle(data)
        expected = 56*57*(2*56+1)/6
        self.assertEqual(self.func(data), expected)
        # sum of cubes of 1, 2, 3, ... n = n**2(n+1)**2/4 = (1+2+...+n)**2
        data1 = list(range(1, 85))
        random.shuffle(data1)
        data2 = [n**3 for n in data1]
        random.shuffle(data2)
        expected = (84**2*85**2)/4
        self.assertEqual(self.func(data1)**2, expected)
        self.assertEqual(self.func(data2), expected)

    def testStartArgument(self):
        # Test that the optional start argument works correctly.
        data = [random.uniform(1, 1000) for _ in range(100)]
        t = self.func(data)
        for start in (42, -23, 1e20):
            self.assertEqual(self.func(data, start), t+start)

    def testFractionSum(self):
        F = Fraction
        # Same denominator (or int).
        data = [F(3, 5), 1, F(4, 5), -F(7, 5), F(9, 5)]
        start = F(1, 5)
        expected = F(3, 1)
        self.assertEqual(self.func(data, start), expected)
        # Different denominators.
        data = [F(9, 4), F(3, 7), 2, -F(2, 5), F(1, 3)]
        start = F(1, 2)
        expected = F(2147, 420)
        self.assertEqual(self.func(data, start), expected)

    def testDecimalSum(self):
        D = Decimal
        data = [D('0.7'), 3, -D('4.3'), D('2.9'), D('3.6')]
        start = D('1.5')
        expected = D('7.4')
        self.assertEqual(self.func(data, start), expected)

    def testFloatSubclass(self):
        class MyFloat(float):
            def __add__(self, other):
                return MyFloat(super().__add__(other))
            __radd__ = __add__
        data = [1.25, 2.5, 7.25, 1.0, 0.0, 3.5, -4.5, 2.25]
        data = map(MyFloat, data)
        expected = MyFloat(13.25)
        actual = self.func(data)
        self.assertEqual(actual, expected)
        self.assertTrue(isinstance(actual, MyFloat))

    def testFloatSum(self):
        data = [2.77, 4.23, 1.91, 0.35, 4.01, 0.57, -4.15, 8.62]
        self.assertEqual(self.func(data), 18.31)
        data = [2.3e19, 7.8e18, 1.0e20, 3.5e19, 7.2e19]
        self.assertEqual(self.func(data), 2.378e20)


class SumTortureTest(NumericTestCase):
    def testTorture(self):
        # Variants on Tim Peters' torture test for sum.
        func = calcstats.sum
        self.assertEqual(func([1, 1e100, 1, -1e100]*10000), 20000.0)
        self.assertEqual(func([1e100, 1, 1, -1e100]*10000), 20000.0)
        self.assertApproxEqual(
            func([1e-100, 1, 1e-100, -1]*10000), 2.0e-96, rel=1e-15, tol=0)


# === Test products ===

class RunningProductTest(unittest.TestCase, TestConsumerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = calcstats.running_product

    def testProduct(self):
        cr = self.func()
        data = [3, 5, 1, -2, -0.5, 0.75]
        expected = [3, 15, 15, -30, 15.0, 11.25]
        assert len(data)==len(expected)
        for x, y in zip(data, expected):
            self.assertEqual(cr.send(x), y)

    def testProductStart(self):
        start = 1.275
        cr = self.func(start)
        data = [2, 5.5, -4, 1.0, -0.25, 1.25]
        expected = [2, 11.0, -44.0, -44.0, 11.0, 13.75]
        assert len(data)==len(expected)
        for x, y in zip(data, expected):
            self.assertEqual(cr.send(x), start*y)

    def testFractions(self):
        F = Fraction
        data = [F(3, 5), 2, F(1, 4), F(5, 3)]
        expected = [F(3, 5), F(6, 5), F(6, 20), F(1, 2)]
        assert len(data)==len(expected)
        start = F(1, 7)
        rs = self.func(start)
        for f, y in zip(data, expected):
            x = rs.send(f)
            self.assertEqual(x, start*y)
            self.assertTrue(isinstance(x, Fraction))

    def testDecimals(self):
        D = Decimal
        data = [D('0.4'), 4, D('2.5'), D('1.7')]
        expected = [D('0.4'), D('1.6'), D('4.0'), D('6.8')]
        assert len(data)==len(expected)
        start = D('1.35')
        rs = self.func(start)
        for d, y in zip(data, expected):
            x = rs.send(d)
            self.assertEqual(x, start*y)
            self.assertTrue(isinstance(x, Decimal))


class ProductTest(NumericTestCase, UnivariateMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = calcstats.product

    def testEmptyData(self):
        # Override UnivariateMixin method.
        for empty in ([], (), iter([])):
            self.assertEqual(self.func(empty), 1)
            for start in (Fraction(23, 42), Decimal('3.456'), 123.456):
                self.assertEqual(self.func(empty, start), start)

    def testStartArgument(self):
        # Test that the optional start argument works correctly.
        data = [random.uniform(-10, 10) for _ in range(100)]
        t = self.func(data)
        for start in (2.1, -3.7, 1e10):
            self.assertApproxEqual(self.func(data, start), t*start, rel=2e-15)

    def testFractionProduct(self):
        F = Fraction
        data = [F(9, 4), F(3, 7), 2, -F(2, 5), F(1, 3), -F(1, 3)]
        start = F(1, 2)
        expected = F(3, 70)
        self.assertEqual(self.func(data, start), expected)

    def testDecimalProduct(self):
        D = Decimal
        data = [D('0.5'), 8, -D('4.75'), D('2.0'), D('3.25'), -D('5.0')]
        start = D('1.5')
        expected = D('926.25')
        self.assertEqual(self.func(data, start), expected)

    def testFloatSubclass(self):
        class MyFloat(float):
            def __mul__(self, other):
                return MyFloat(super().__mul__(other))
            __rmul__ = __mul__
        data = [2.5, 4.25, -1.0, 3.5, -0.5, 0.25]
        data = map(MyFloat, data)
        expected = MyFloat(4.6484375)
        actual = self.func(data)
        self.assertEqual(actual, expected)
        self.assertTrue(isinstance(actual, MyFloat))

    def testFloatProduct(self):
        data = [0.71, 4.10, 0.18, 2.47, 3.11, 0.79, 1.52, 2.31]
        expected = 11.1648967698  # Calculated with HP-48GX.
        self.assertApproxEqual(self.func(data), 11.1648967698, tol=1e-10)
        data = [2, 3, 5, 10, 0.25, 0.5, 2.5, 1.5, 4, 0.2]
        self.assertEqual(self.func(data), 112.5)


# === Test means ===

class RunningMeanTest(unittest.TestCase, TestConsumerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = calcstats.running_mean

    def testFloats(self):
        cr = self.func()
        data = [3, 5, 0, -1, 0.5, 1.75]
        expected = [3, 4.0, 8/3, 1.75, 1.5, 9.25/6]
        assert len(data)==len(expected)
        for x, y in zip(data, expected):
            self.assertEqual(cr.send(x), y)

    def testFractions(self):
        F = Fraction
        data = [F(3, 5), F(1, 5), F(1, 3), 3, F(5, 3)]
        expected = [F(3, 5), F(2, 5), F(17, 45), F(31, 30), F(29, 25)]
        assert len(data)==len(expected)
        rs = self.func()
        for f, y in zip(data, expected):
            x = rs.send(f)
            self.assertEqual(x, y)
            self.assertTrue(isinstance(x, Fraction))

    def testDecimals(self):
        D = Decimal
        data = [D('3.4'), 2, D('3.9'), -D('1.3'), D('4.2')]
        expected = [D('3.4'), D('2.7'), D('3.1'), D('2.0'), D('2.44')]
        assert len(data)==len(expected)
        rs = self.func()
        for d, y in zip(data, expected):
            x = rs.send(d)
            self.assertEqual(x, y)
            self.assertTrue(isinstance(x, Decimal))


class MeanTest(NumericTestCase, UnivariateMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = calcstats.mean
        self.data = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
        self.expected = 5.5

    def setUp(self):
        random.shuffle(self.data)

    def testSeq(self):
        self.assertApproxEqual(self.func(self.data), self.expected)

    def testShiftedData(self):
        # Shifting data shouldn't change the mean.
        data = [x + 1e9 for x in self.data]
        expected = self.expected + 1e9
        assert expected != 1e9
        self.assertApproxEqual(self.func(data), expected)

    def testIter(self):
        self.assertApproxEqual(self.func(iter(self.data)), self.expected)

    def testSingleton(self):
        for x in self.data:
            self.assertEqual(self.func([x]), x)

    def testDoubling(self):
        # Average of [a,b,c...z] should be same as for [a,a,b,b,c,c...z,z].
        data = [random.random() for _ in range(1000)]
        a = self.func(data)
        b = self.func(data*2)
        self.assertApproxEqual(a, b)

    def testAddMean(self):
        # Adding the mean to a data set shouldn't change the mean.
        data = [random.random() for _ in range(1000)]
        a = self.func(data)
        data.extend([a]*123)
        random.shuffle(data)
        b = self.func(data)
        self.assertApproxEqual(a, b, tol=1e-15)


# === Test variances and standard deviations ===

class WelfordTest(NumericTestCase, TestConsumerMixin):
    # Expected results were either calculated by hand, or using a HP-48GX
    # calculator with the RPL program: « Σ+ PVAR NΣ * »

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = calcstats.welford

    def testFloats(self):
        cr = self.func()
        data = [2.5, 3.25, 5, -0.5, 1.75, 2.5, 3.5]
        expected = [0.0, 0.28125, 3.29166666666, 15.796875, 16.325,
                    16.3333333333, 17.3392857143]
        assert len(data)==len(expected)
        for x, y in zip(data, expected):
            self.assertApproxEqual(cr.send(x), y, tol=1e-10)

    def testFractions(self):
        cr = self.func()
        F = Fraction
        data = [F(2), F(3), F(4), F(5), F(6)]
        expected = [F(0), F(1, 2), F(2, 1), F(5, 1), F(10, 1)]
        assert len(data)==len(expected)
        for f, y in zip(data, expected):
            x = cr.send(f)
            self.assertEqual(x, y)
            self.assertTrue(isinstance(x, Fraction))

    def testDecimals(self):
        D = Decimal
        cr = self.func()
        data = [D(3), D(5), D(4), D(3), D(5), D(4)]
        expected = [D(0), D(2), D(2), D('2.75'), D(4), D(4)]
        assert len(data)==len(expected)
        for d, y in zip(data, expected):
            x = cr.send(d)
            self.assertEqual(x, y)
            self.assertTrue(isinstance(x, Decimal))
        x = cr.send(D(-2))
        self.assertApproxEqual(x, D('34.8571428571'), tol=D('1e-10'))


class PrivateVarTest(unittest.TestCase):
    # Test the _variance private function.

    def test_enough_points(self):
        # Test that _variance succeeds if N-p is positive.
        for N in range(1, 8):
            for p in range(N):
                data = range(N)
                assert len(data) > p
                _ = calcstats._variance(data, 2.5, p)

    def test_too_few_points(self):
        # Test that _variance fails if N-p is too low.
        StatsError = calcstats.StatsError
        var = calcstats._variance
        for p in range(5):
            for N in range(p):
                data = range(N)
                assert len(data) <= p
                self.assertRaises(StatsError, var, data, 2.5, p)

    def test_error_msg(self):
        # Test that the error message is correct.
        try:
            calcstats._variance([4, 6, 8], 2.5, 5)
        except calcstats.StatsError as e:
            self.assertEqual(
                e.args[0], 'at least 6 items are required but only got 3'
                )
        else:
            self.fail('expected StatsError exception did not get raised')

    def test_float_sequence(self):
        data = [3.5, 5.5, 4.0, 2.5, 2.0]
        assert sum(data)/len(data) == 3.5  # mean
        actual = calcstats._variance(data, 3.5, 3)
        expected = (0 + 2**2 + 0.5**2 + 1 + 1.5**2)/2
        self.assertEqual(actual, expected)

    def test_fraction_sequence(self):
        F = Fraction
        data = [F(2, 5), F(3, 4), F(1, 4), F(2, 3)]
        assert sum(data)/len(data) == F(31, 60)  # mean
        actual = calcstats._variance(data, F(31, 15), 2)
        expected = (F(7,60)**2 + F(14,60)**2 + F(16,60)**2 + F(9,60)**2)/2
        self.assertEqual(actual, expected)

    def test_decimal_sequence(self):
        D = Decimal
        data = [D(2), D(2), D(5), D(7)]
        assert sum(data)/len(data) == D(4)  # mean
        actual = calcstats._variance(data, D(4), 2)
        expected = (D(2)**2 + D(2)**2 + D(1)**2 + D(3)**2)/2
        self.assertEqual(actual, expected)


class ExactVarianceTest(unittest.TestCase):
    # Exact tests for variance and friends.
    def testExactVariance1(self):
        data = [1, 2, 3]
        assert calcstats.mean(data) == 2
        self.assertEqual(calcstats.pvariance(data), 2/3)
        self.assertEqual(calcstats.variance(data), 1.0)
        self.assertEqual(calcstats.pstdev(data), math.sqrt(2/3))
        self.assertEqual(calcstats.stdev(data), 1.0)

    def testExactVariance2(self):
        data = [1, 1, 1, 2, 3, 7]
        assert calcstats.mean(data) == 2.5
        self.assertEqual(calcstats.pvariance(data), 165/36)
        self.assertEqual(calcstats.variance(data), 165/30)
        self.assertEqual(calcstats.pstdev(data), math.sqrt(165/36))
        self.assertEqual(calcstats.stdev(data), math.sqrt(165/30))

    def testExactVarianceFrac(self):
        data = [Fraction(100), Fraction(200), Fraction(600)]
        assert calcstats.mean(data) == Fraction(300)
        self.assertEqual(calcstats.pvariance(data), Fraction(420000, 9))
        self.assertEqual(calcstats.variance(data), Fraction(70000))
        self.assertEqual(calcstats.pstdev(data), math.sqrt(420000/9))
        self.assertEqual(calcstats.stdev(data), math.sqrt(70000))

    def testExactVarianceDec(self):
        data = [Decimal('1.1'), Decimal('1.2'), Decimal('1.9')]
        assert calcstats.mean(data) == Decimal('1.4')
        self.assertEqual(calcstats.pvariance(data), Decimal('1.14')/9)
        self.assertEqual(calcstats.variance(data), Decimal('0.19'))
        self.assertEqual(calcstats.pstdev(data), math.sqrt(1.14/9))
        self.assertEqual(calcstats.stdev(data), math.sqrt(0.19))


class VarianceUnbiasedTest(NumericTestCase):
    # Test that variance is unbiased.
    tol = 5e-11
    rel = 5e-16

    def testUnbiased(self):
        # Test that variance is unbiased with known data.
        data = [1, 1, 2, 5]  # Don't give data too many items or this
                             # will be way too slow!
        assert calcstats.mean(data) == 2.25
        assert calcstats.pvariance(data) == 2.6875
        samples = self.get_all_samples(data)
        sample_variances = [calcstats.variance(sample) for sample in samples]
        self.assertEqual(calcstats.mean(sample_variances), 2.6875)

    def testRandomUnbiased(self):
        # Test that variance is unbiased with random data.
        data = [random.uniform(-100, 1000) for _ in range(5)]
        samples = self.get_all_samples(data)
        pvar = calcstats.pvariance(data)
        sample_variances = [calcstats.variance(sample) for sample in samples]
        self.assertApproxEqual(calcstats.mean(sample_variances), pvar)

    def get_all_samples(self, data):
        """Return a generator that returns all permutations with
        replacement of the given data."""
        return itertools.chain(
            *(itertools.product(data, repeat=n) for n in
            range(2, len(data)+1)))


class PVarianceTest(NumericTestCase, UnivariateMixin):
    # Test population variance.
    # This will be subclassed by variance and [p]stdev.
    tol = 1e-11

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = calcstats.pvariance
        # Test data for test_main, test_shift:
        self.data = [4.0, 7.0, 13.0, 16.0]
        self.expected = 22.5  # Exact population variance of self.data.
        # If you duplicate each data point, the variance will scale by
        # this value:
        self.dup_scale_factor = 1.0

    def setUp(self):
        random.shuffle(self.data)

    def get_allowed_kinds(self):
        kinds = super().get_allowed_kinds()
        return [kind for kind in kinds if hasattr(kind, '__len__')]

    def test_main(self):
        # Test that pvariance calculates the correct result.
        self.assertEqual(self.func(self.data), self.expected)

    def test_shift(self):
        # Shifting the data by a constant amount should not affect
        # the variance.
        for shift in (1e2, 1e6, 1e9):
            data = [x + shift for x in self.data]
            self.assertEqual(self.func(data), self.expected)

    def test_equal_data(self):
        # If the data is constant, the variance should be zero.
        self.assertEqual(self.func([42]*10), 0)

    def testDuplicate(self):
        # Test that the variance behaves as expected when you duplicate
        # each data point [a,b,c,...] -> [a,a,b,b,c,c,...]
        data = [random.uniform(-100, 500) for _ in range(20)]
        expected = self.func(data)*self.dup_scale_factor
        actual = self.func(data*2)
        self.assertApproxEqual(actual, expected)

    def testDomainError(self):
        # Domain error exception reported by Geremy Condra.
        data = [0.123456789012345]*10000
        # All the items are identical, so variance should be exactly zero.
        # We allow some small round-off error.
        self.assertApproxEqual(self.func(data), 0.0, tol=5e-17)

    def testSingleton(self):
        # Population variance of a single value is always zero.
        for x in self.data:
            self.assertEqual(self.func([x]), 0)

    def testMeanArgument(self):
        # Variance calculated with the given mean should be the same
        # as that calculated without the mean.
        data = [random.random() for _ in range(15)]
        m = calcstats.mean(data)
        expected = self.func(data, m=None)
        self.assertEqual(self.func(data, m=m), expected)


class VarianceTest(PVarianceTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = calcstats.variance
        self.expected = 30.0  # Exact sample variance of self.data.
        # Scaling factor when you duplicate each data point:
        self.dup_scale_factor = (2*20-2)/(2*20-1)

    def testSingleData(self):
        # Override mixin test.
        self.assertRaises(calcstats.StatsError, self.func, [23])

    # Note that testSingleData and testSingleton are not redundant tests!
    # Although they both end up doing the same thing, they are both needed
    # to override tests which do different things in the superclasses.

    def testSingleton(self):
        # Override pvariance test.
        self.assertRaises(calcstats.StatsError, self.func, [42])


class PStdevTest(PVarianceTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = calcstats.pstdev
        self.expected = math.sqrt(self.expected)
        self.dup_scale_factor = math.sqrt(self.dup_scale_factor)


class StdevTest(VarianceTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = calcstats.stdev
        self.expected = math.sqrt(self.expected)
        self.dup_scale_factor = math.sqrt(self.dup_scale_factor)


class VarianceComparedTest(NumericTestCase):
    # Compare variance calculations with results calculated using
    # HP-48GX calculator and R.
    tol = 1e-7
    def __init__(self, *args, **kwargs):
        NumericTestCase.__init__(self, *args, **kwargs)
        self.data = (
            list(range(1, 11)) + list(range(1000, 1201)) +
            [0, 3, 7, 23, 42, 101, 111, 500, 567]
            )

    def setUp(self):
        random.shuffle(self.data)

    def test_pvariance(self):
        # Compare the calculated population variance against the result
        # calculated by the HP-48GX calculator.
        self.assertApproxEqual(calcstats.pvariance(self.data), 88349.2408884)

    def test_variance(self):
        # As above, for sample variance.
        self.assertApproxEqual(calcstats.variance(self.data), 88752.6620797)

    def test_pstdev(self):
        # As above, for population standard deviation.
        self.assertApproxEqual(calcstats.pstdev(self.data), 297.236002006)

    def test_stdev(self):
        # As above, for sample standard deviation.
        self.assertApproxEqual(calcstats.stdev(self.data), 297.913850097)

    def testCompareVarianceWithR(self):
        # Compare against a result calculated with R:
        #       > x <- c(seq(1, 10), seq(1000, 1200))
        #       > var(x)
        #       [1] 57563.55
        data = list(range(1, 11)) + list(range(1000, 1201))
        expected = 57563.55
        self.assertApproxEqual(calcstats.variance(data), expected, tol=1e-3)
            # The expected value from R looks awfully precise... does R
            # round figures? I don't think it is the exact value, as
            # my HP-48GX calculator returns 57563.5502144.

    def testCompareStdevWithR(self):
        # Compare with a result calculated by R.
        data = list(range(1, 11)) + list(range(1000, 1201))
        expected = 239.9241
        self.assertApproxEqual(calcstats.stdev(data), expected, tol=1e-4)


class VarianceUniformData(unittest.TestCase):
    # Compare variances against the expected value for uniformly distributed
    # data [0, 1, 2, 3, 4, 5, ...]
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.data = list(range(10000))
        # Exact value for population variance:
        self.expected = (10000**2 - 1)/12

    def setUp(self):
        random.shuffle(self.data)

    def test_pvariance(self):
        # Compare the calculated population variance against the exact result.
        self.assertEqual(calcstats.pvariance(self.data), self.expected)

    def test_variance(self):
        # Compare the calculated sample variance against the exact result.
        expected = self.expected*10000/(10000-1)
        self.assertEqual(calcstats.variance(self.data), expected)

    def test_pstdev(self):
        # Compare the calculated population std dev against the exact result.
        expected = math.sqrt(self.expected)
        self.assertEqual(calcstats.pstdev(self.data), expected)

    def test_stdev(self):
        # Compare the calculated sample variance against the exact result.
        expected = math.sqrt(self.expected*10000/(10000-1))
        self.assertEqual(calcstats.stdev(self.data), expected)


class PVarianceDupsTest(NumericTestCase):
    tol=1e-12

    def testManyDuplicates(self):
        # Start with 1000 normally distributed data points.
        data = [random.gauss(7.5, 5.5) for _ in range(1000)]
        expected = calcstats.pvariance(data)
        # We expect the calculated variance to be close to the exact result
        # for the variance of normal data, namely 5.5**2, but because the
        # data was generated randomly, it might not be. But if it isn't,
        # it doesn't matter.
        #
        # Duplicating the data points should keep the variance the same.
        for n in (3, 5, 10, 20, 30):
            d = data*n
            actual = calcstats.pvariance(d)
            self.assertApproxEqual(actual, expected)
        # FIXME -- we should test this with LOTS of duplicates, but that
        # will probably have to wait for support for iterator data streams.


class TestAgainstCompFormulaP(TestCompPVariance):
    """Test that the population variance succeeds in calculations that the
    so-called 'computational formula of the variance' fails at.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = calcstats.pvariance

    def test_shifted_variance(self):
        data = self.shifted_data()
        self.assertEqual(self.func(data), self.expected)


class TestAgainstCompFormula(TestCompVariance):
    """Test that the sample variance succeeds in calculations that the
    so-called 'computational formula of the variance' fails at.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = calcstats.variance

    def test_shifted_variance(self):
        data = self.shifted_data()
        self.assertEqual(self.func(data), self.expected*400/499)


# === Test other statistics functions ===

class MinmaxTest(unittest.TestCase):
    """Tests for minmax function."""
    data = list(range(100))
    expected = (0, 99)

    def key(self, n):
        # This must be a monotomically increasing function.
        return n*33 - 11

    def setUp(self):
        self.minmax = calcstats.minmax
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

    def testInPlaceModification(self):
        # Test that minmax does not modify its input data.
        data = [3, 0, 5, 1, 7, 2, 9, 4, 8, 6]
        # We wish to detect functions that modify the data in place by
        # sorting, which we can't do if the data is already sorted.
        assert data != sorted(data)
        saved = data[:]
        assert data is not saved
        result = self.minmax(data)
        self.assertEqual(result, (0, 9))
        self.assertEqual(data, saved, "data has been modified")

    def testTypes(self):
        class MyList(list): pass
        class MyTuple(tuple): pass
        def generator(seq):
            return (x for x in seq)
        for kind in (list, MyList, tuple, MyTuple, generator, iter):
            data = kind(self.data)
            self.assertEqual(self.minmax(data), self.expected)

    def testAbsKey(self):
        data = [-12, -8, -4, 2, 6, 10]
        random.shuffle(data)
        self.assertEqual(self.minmax(data, key=abs), (2, -12))
        random.shuffle(data)
        self.assertEqual(self.minmax(*data, key=abs), (2, -12))



# === Run tests ===

class DocTests(unittest.TestCase):
    def testMyDocTests(self):
        import doctest
        failed, tried = doctest.testmod()
        self.assertTrue(tried > 0)
        self.assertTrue(failed == 0)

    def testStatsDocTests(self):
        import doctest
        failed, tried = doctest.testmod(calcstats)
        self.assertTrue(tried > 0)
        self.assertTrue(failed == 0)



if __name__ == '__main__':
    unittest.main()

