"""Unit tests for calcstats.py"""

from decimal import Decimal
from fractions import Fraction

import collections
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
    """Test whether x is approximately equal to y, using an absolute error
    of tol and/or a relative error of rel, whichever is bigger.

    Pass None as either tol or rel to ignore that test; if both are None,
    the test performed is an exact equality test.

    tol and rel must be either None or a positive, finite number, otherwise
    the behaviour is undefined.
    """
    if tol is rel is None:
        # Fall back on exact equality.
        return x == y
    # NANs are never equal to anything, approximately or otherwise.
    if math.isnan(x) or math.isnan(y):
        # FIXME Signalling NANs should raise an exception.
        return False
    # Infinities are approximately equal if they have the same sign.
    if math.isinf(x) or math.isinf(y):
        return x == y
    # If we get here, both x and y are finite, and likewise delta.
    delta = abs(x - y)
    tests = []
    if tol is not None:
        tests.append(tol)
    if rel is not None:
        tests.append(rel*max(abs(x), abs(y)))
    assert tests
    return delta <= max(tests)


# Generic test suite subclass
# ---------------------------
# We prefer this for testing numeric values that may not be exactly equal.
# Avoid using TestCase.almost_equal, because it sucks :)

USE_DEFAULT = object()
class NumericTestCase(unittest.TestCase):
    # By default, we expect exact equality, unless overridden.
    tol = None
    rel = None

    def assertApproxEqual(
        self, actual, expected, tol=USE_DEFAULT, rel=USE_DEFAULT, msg=None
        ):
        if tol is USE_DEFAULT: tol = self.tol
        if rel is USE_DEFAULT: rel = self.rel
        if (isinstance(actual, collections.Sequence) and
        isinstance(expected, collections.Sequence)):
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
        # Create the standard error message, starting with the common part,
        # which comes at the end.
        abs_err = abs(actual - expected)
        rel_err1 = abs_err/abs(expected) if expected else float('inf')
        rel_err2 = abs_err/abs(actual) if actual else float('inf')
        rel_err = min(rel_err1, rel_err2)
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


# Here we test the test infrastructure itself.

class ApproxIntegerTest(unittest.TestCase):
    # Test the approx_equal function with ints.

    def _equality_tests(self, x, y):
        """Test ways of spelling 'exactly equal'."""
        return (approx_equal(x, y, tol=None, rel=None),
                approx_equal(y, x, tol=None, rel=None),
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
        self.assertTrue(approx_equal(-42, -43, tol=1, rel=None))
        self.assertTrue(approx_equal(15, 16, tol=2, rel=None))
        self.assertFalse(approx_equal(23, 27, tol=3, rel=None))

    def testRelative(self):
        # Test approximate equality with a relative error.
        self.assertTrue(approx_equal(100, 119, tol=None, rel=0.2))
        self.assertTrue(approx_equal(119, 100, tol=None, rel=0.2))
        self.assertFalse(approx_equal(100, 130, tol=None, rel=0.2))
        self.assertFalse(approx_equal(130, 100, tol=None, rel=0.2))

class ApproxFractionTest(unittest.TestCase):
    # Test the approx_equal function with Fractions.

    def testExactlyEqual(self):
        # Test that equal values are equal and unequal values are unequal.
        F = Fraction
        values = [-F(1, 2), F(0), F(5, 3), F(9, 7), F(35, 36)]
        for x in values:
            self.assertTrue(
                approx_equal(x, x, tol=None, rel=None),
                'equality failure for x=%r' % x
                )
            self.assertFalse(
                approx_equal(x, x+1, tol=None, rel=None),
                'inequality failure for x=%r' % x
                )

    def testAbsolute(self):
        # Test approximate equality with an absolute error.
        F = Fraction
        aeq = approx_equal
        self.assertTrue(aeq(F(7, 9), F(8, 9), tol=F(1, 9), rel=None))
        self.assertTrue(aeq(F(8, 5), F(7, 5), tol=F(2, 5), rel=None))
        self.assertFalse(aeq(F(6, 8), F(8, 8), tol=F(1, 8), rel=None))

    def testRelative(self):
        # Test approximate equality with a relative error.
        F = Fraction
        aeq = approx_equal
        self.assertTrue(aeq(F(45, 100), F(65, 100), tol=None, rel=F(32, 100)))
        self.assertFalse(aeq(F(23, 50), F(48, 50), tol=None, rel=F(26, 50)))


class ApproxDecimalTest(unittest.TestCase):
    # Test the approx_equal function with Decimals.

    def testExactlyEqual(self):
        # Test that equal values are equal and unequal values are unequal.
        D = Decimal
        values = [D('-23.0'), D(0), D('1.3e-15'), D('3.25'), D('1.7e15')]
        for x in values:
            self.assertTrue(
                approx_equal(x, x, tol=None, rel=None),
                'equality failure for x=%r' % x
                )
            self.assertFalse(
                approx_equal(x, x+1, tol=None, rel=None),
                'inequality failure for x=%r' % x
                )

    def testAbsolute(self):
        # Test approximate equality with an absolute error.
        D = Decimal
        aeq = approx_equal
        self.assertTrue(aeq(D('12.78'), D('12.35'), tol=D('0.43'), rel=None))
        self.assertTrue(aeq(D('35.4'), D('36.2'), tol=D('1.5'), rel=None))
        self.assertFalse(aeq(D('35.3'), D('36.2'), tol=D('0.8'), rel=None))

    def testRelative(self):
        # Test approximate equality with a relative error.
        D = Decimal
        aeq = approx_equal
        self.assertTrue(aeq(D('5.4'), D('6.7'), tol=None, rel=D('0.20')))
        self.assertFalse(aeq(D('5.4'), D('6.7'), tol=None, rel=D('0.19')))

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
                approx_equal(x, x, tol=None, rel=None),
                'equality failure for x=%r' % x
                )
            self.assertFalse(
                approx_equal(x, x+1, tol=None, rel=None),
                'inequality failure for x=%r' % x
                )

    def testAbsolute(self):
        # Test approximate equality with an absolute error.
        self.assertTrue(approx_equal(4.57, 4.54, tol=0.5, rel=None))
        self.assertTrue(approx_equal(4.57, 4.52, tol=0.5, rel=None))
        self.assertTrue(approx_equal(2.3e12, 2.6e12, tol=0.4e12, rel=None))
        self.assertFalse(approx_equal(2.3e12, 2.6e12, tol=0.2e12, rel=None))
        self.assertTrue(approx_equal(1.01e-9, 1.03e-9, tol=0.05e-9, rel=None))
        self.assertTrue(approx_equal(273.5, 263.9, tol=9.7, rel=None))
        self.assertFalse(approx_equal(273.5, 263.9, tol=9.0, rel=None))

    def testRelative(self):
        # Test approximate equality with a relative error.
        self.assertTrue(approx_equal(3.5, 4.1, tol=None, rel=0.147))
        self.assertFalse(approx_equal(3.5, 4.1, tol=None, rel=0.146))
        self.assertTrue(approx_equal(7.2e11, 6.9e11, tol=None, rel=0.042))
        self.assertFalse(approx_equal(7.2e11, 6.9e11, tol=None, rel=0.041))

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
        nzero = math.copysign(0, -1)
        self.assertTrue(approx_equal(nzero, 0.0, tol=1, rel=1))
        self.assertTrue(approx_equal(0.0, nzero, tol=None, rel=None))


class TestNumericTestCase(unittest.TestCase):
    # The formatting routine that generates the error messages is complex
    # enough that it needs its own test.

    def test_error_msg_exact(self):
        # Test the error message generated for exact tests.
        msg = NumericTestCase._make_std_err_msg(0.5, 0.25, None, None, None)
        self.assertEqual(msg,
            "actual value 0.5 is not equal to expected 0.25\n"
            "    absolute error = 0.25\n"
            "    relative error = 0.5"
            )

    def test_error_msg_inexact(self):
        # Test the error message generated for inexact tests.
        msg = NumericTestCase._make_std_err_msg(2.5, 1.25, 0.25, None, None)
        self.assertEqual(msg,
            "actual value 2.5 differs from expected 1.25\n"
            "    by more than tol=0.25\n"
            "    absolute error = 1.25\n"
            "    relative error = 0.5"
            )
        msg = NumericTestCase._make_std_err_msg(1.5, 2.5, None, 0.25, None)
        self.assertEqual(msg,
            "actual value 1.5 differs from expected 2.5\n"
            "    by more than rel=0.25\n"
            "    absolute error = 1.0\n"
            "    relative error = 0.4"
            )
        msg = NumericTestCase._make_std_err_msg(2.5, 4.0, 0.5, 0.25, None)
        self.assertEqual(msg,
            "actual value 2.5 differs from expected 4.0\n"
            "    by more than tol=0.5 and rel=0.25\n"
            "    absolute error = 1.5\n"
            "    relative error = 0.375"
            )

    def test_error_msg_sequence(self):
        # Test the error message generated for sequence tests.
        msg = NumericTestCase._make_std_err_msg(2.5, 4.0, 0.5, 0.25, 7)
        self.assertEqual(msg,
            "numeric sequences first differs at index 7.\n"
            "actual value 2.5 differs from expected 4.0\n"
            "    by more than tol=0.5 and rel=0.25\n"
            "    absolute error = 1.5\n"
            "    relative error = 0.375"
            )

    def testNumericTestCaseIsTestCase(self):
        # Ensure that NumericTestCase actually is a TestCase.
        self.assertTrue(issubclass(NumericTestCase, unittest.TestCase))



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


# === Test utility functions ===

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


# === Test running sum, product and mean ===

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


# === Test sum, product and mean ===

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

    def testTypeOfDataCollection(self):
        # Test that the type of iterable data doesn't effect the result.
        class MyList(list):
            pass
        class MyTuple(tuple):
            pass
        def generator(data):
            return (obj for obj in data)
        data = range(1, 16, 2)
        expected = self.func(data)
        for kind in (list, tuple, iter, MyList, MyTuple, generator):
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
            func([1e-100, 1, 1e-100, -1]*10000), 2.0e-96, rel=1e-15, tol=None)


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

    def testBigData(self):
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

if __name__ == '__main__':
    unittest.main()

