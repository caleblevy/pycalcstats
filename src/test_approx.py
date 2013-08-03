"""Numeric approximated equal comparisons and unit testing.

When doing numeric work, especially with floats, exact equality is often not
what you want. Due to round-off error, it is often a bad idea to try to
compare floats with equality. Instead the usual procedure is to test them
with some (hopefully small!) allowance for error.

The ``approx_equal`` function allows you to specify either an absolute error
tolerance, or a relative error, or both.

Absolute error tolerances are simple, but you need to know the magnitude of
the quantities being compared:

>>> approx_equal(12.345, 12.346, tol=1e-3)
True
>>> approx_equal(12.345e6, 12.346e6, tol=1e-3)  # tol is unreasonably small.
False

Relative errors are more suitable when the values you are comparing can vary
in magnitude:

>>> approx_equal(12.345, 12.346, rel=1e-4)
True
>>> approx_equal(12.345e6, 12.346e6, rel=1e-4)
True

but a naive implementation of relative error testing can run into trouble
around zero.

If you supply both an absolute tolerance and a relative error, the comparison
succeeds if either individual test succeeds:

>>> approx_equal(12.345e6, 12.346e6, tol=1e-3, rel=1e-4)
True

For unit-testing, a TestCase subclass ``NumericTestCase`` is provided. In
addition to the standard ``TestCase.assertAlmostEqual``, a method
``assertApproxEqual`` is provided. See the class for further details.


This is a self-testing module. From the commandline, call:

    $ python3 -m test_approx

to run the self-tests.
"""

__all__ = ['approx_equal', 'NumericTestCase']

from decimal import Decimal
from fractions import Fraction

import collections
import doctest
import math
import unittest


# === Helper functions ===

def _calc_errors(actual, expected):
    """Return the absolute and relative errors between two numbers.

    >>> _calc_errors(100, 75)
    (25, 0.25)
    >>> _calc_errors(100, 100)
    (0, 0.0)

    Returns the (absolute error, relative error) between the two arguments.
    """
    base = max(abs(actual), abs(expected))
    abs_err = abs(actual - expected)
    rel_err = abs_err/base if base else float('inf')
    return (abs_err, rel_err)


# === Approximately equal to ===

def approx_equal(x, y, tol=1e-12, rel=1e-7):
    """approx_equal(x, y [, tol [, rel]]) => True|False

    Return True if numbers x and y are approximately equal, to within some
    margin of error, otherwise return False. Numbers which compare equal
    will also compare approximately equal.

    x is approximately equal to y if the difference between them is less than
    an absolute error tol or a relative error rel, whichever is bigger.

    If given, both tol and rel must be finite, non-negative numbers. If not
    given, default values are tol=1e-12 and rel=1e-7.

    >>> approx_equal(1.2589, 1.2587, tol=0.0003, rel=0)
    True
    >>> approx_equal(1.2589, 1.2587, tol=0.0001, rel=0)
    False

    Absolute error is defined as abs(x-y); if that is less than or equal to
    tol, x and y are considered approximately equal.

    Relative error is defined as abs((x-y)/x) or abs((x-y)/y), whichever is
    smaller, provided x or y are not zero. If that figure is less than or
    equal to rel, x and y are considered approximately equal.

    Complex numbers are not directly supported. If you wish to compare to
    complex numbers, extract their real and imaginary parts and compare them
    individually.

    NANs always compare unequal, even with themselves. Infinities compare
    approximately equal if they have the same sign (both positive or both
    negative). Infinities with different signs compare unequal; so do
    comparisons of infinities with finite numbers.
    """
    if tol < 0 or rel < 0:
        raise ValueError('error tolerances must be non-negative')
    # NANs are never equal to anything, approximately or otherwise.
    if math.isnan(x) or math.isnan(y):
        return False
    # Numbers which compare equal also compare approximately equal.
    if x == y:
        # This includes the case of two infinities with the same sign.
        return True
    if math.isinf(x) or math.isinf(y):
        # This includes the case of two infinities of opposite sign, or
        # one infinity and one finite number.
        return False
    # Two finite numbers.
    actual_error = abs(x - y)
    allowed_error = max(tol, rel*max(abs(x), abs(y)))
    return actual_error <= allowed_error


# === Unit test helper ===

# We prefer this for testing numeric values that may not be exactly equal,
# and avoid using TestCase.almost_equal, because it sucks :-)

class NumericTestCase(unittest.TestCase):
    # By default, we expect exact equality, unless overridden.
    tol = rel = 0

    def assertApproxEqual(
            self, actual, expected, tol=None, rel=None, msg=None
            ):
        """Test passes if ``actual`` and ``expected`` are approximately equal.

        This test passes if ``actual`` and ``expected`` are equal to
        within ``tol``, an absolute error, or ``rel``, a relative error.

        If either ``tol`` or ``rel`` are None or not given, they default to
        test attributes of the same name (by default, 0).

        The objects may be either numbers, or sequences of numbers. Sequences
        are tested element-by-element.

        >>> class MyTest(NumericTestCase):
        ...     def testNumber(self):
        ...         x = 1.0/6
        ...         y = sum([x]*6)
        ...         self.assertApproxEqual(y, 1.0, tol=1e-15)
        ...     def testSequence(self):
        ...         a = [1.001, 1.001e-10, 1.001e10]
        ...         b = [1.0, 1e-10, 1e10]
        ...         self.assertApproxEqual(a, b, rel=1e-3)
        ...
        >>> import unittest
        >>> suite = unittest.TestLoader().loadTestsFromTestCase(MyTest)
        >>> unittest.TextTestRunner().run(suite)
        <unittest.runner.TextTestResult run=2 errors=0 failures=0>

        """
        if tol is None:
            tol = self.tol
        if rel is None:
            rel = self.rel
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
            '  by more than tol=%r and rel=%r\n'
            '  -> absolute error = %r\n'
            '  -> relative error = %r'
            )
        if idx is not None:
            header = 'numeric sequences first differ at index %d.\n' % idx
            template = header + template
        # Calculate actual errors:
        abs_err, rel_err = _calc_errors(actual, expected)
        return template % (actual, expected, tol, rel, abs_err, rel_err)


# === Tests for approx_equal ===

class ApproxEqualSymmetryTest(unittest.TestCase):
    # Test symmetry of approx_equal.

    def testRelativeSymmetry(self):
        # Check that approx_equal treats relative error symmetrically.
        # (a-b)/a is usually not equal to (a-b)/b. Ensure that this
        # doesn't matter.
        #
        #   Note: the reason for this test is that an early version
        #   of approx_equal was not symmetric. A relative error test
        #   would pass, or fail, depending on which value was passed
        #   as the first argument.
        #
        args1 = [2456, 37.8, -12.45, Decimal('2.54'), Fraction(17, 54)]
        args2 = [2459, 37.2, -12.41, Decimal('2.59'), Fraction(15, 54)]
        assert len(args1) == len(args2)
        for a, b in zip(args1, args2):
            self.do_relative_symmetry(a, b)

    def do_relative_symmetry(self, a, b):
        a, b = min(a, b), max(a, b)
        assert a < b
        delta = b - a  # The absolute difference between the values.
        rel_err1, rel_err2 = abs(delta/a), abs(delta/b)
        # Choose an error margin halfway between the two.
        rel = (rel_err1 + rel_err2)/2
        # Now see that values a and b compare approx equal regardless of
        # which is given first.
        self.assertTrue(approx_equal(a, b, tol=0, rel=rel))
        self.assertTrue(approx_equal(b, a, tol=0, rel=rel))

    def testSymmetry(self):
        # Test that approx_equal(a, b) == approx_equal(b, a)
        args = [-23, -2, 5, 107, 93568]
        delta = 2
        for x in args:
            for type_ in (int, float, Decimal, Fraction):
                x = type_(x)*100
                y = x + delta
                r = abs(delta/max(x, y))
                # There are five cases to check:
                # 1) actual error <= tol, <= rel
                self.do_symmetry_test(x, y, tol=delta, rel=r)
                self.do_symmetry_test(x, y, tol=delta+1, rel=2*r)
                # 2) actual error > tol, > rel
                self.do_symmetry_test(x, y, tol=delta-1, rel=r/2)
                # 3) actual error <= tol, > rel
                self.do_symmetry_test(x, y, tol=delta, rel=r/2)
                # 4) actual error > tol, <= rel
                self.do_symmetry_test(x, y, tol=delta-1, rel=r)
                self.do_symmetry_test(x, y, tol=delta-1, rel=2*r)
                # 5) exact equality test
                self.do_symmetry_test(x, x, tol=0, rel=0)
                self.do_symmetry_test(x, y, tol=0, rel=0)

    def do_symmetry_test(self, a, b, tol, rel):
        template = "approx_equal comparisons don't match for %r"
        flag1 = approx_equal(a, b, tol, rel)
        flag2 = approx_equal(b, a, tol, rel)
        self.assertEqual(flag1, flag2, template.format((a, b, tol, rel)))


class ApproxEqualExactTest(unittest.TestCase):
    # Test the approx_equal function with exactly equal values.
    # Equal values should compare as approximately equal.
    # Test cases for exactly equal values, which should compare approx
    # equal regardless of the error tolerances given.

    def do_exactly_equal_test(self, x, tol, rel):
        result = approx_equal(x, x, tol=tol, rel=rel)
        self.assertTrue(result, 'equality failure for x=%r' % x)
        result = approx_equal(-x, -x, tol=tol, rel=rel)
        self.assertTrue(result, 'equality failure for x=%r' % -x)

    def testExactlyEqualInts(self):
        # Test that equal int values are exactly equal.
        for n in [42, 19740, 14974, 230, 1795, 700245, 36587]:
            self.do_exactly_equal_test(n, 0, 0)

    def testExactlyEqualFloats(self):
        # Test that equal float values are exactly equal.
        for x in [0.42, 1.9740, 1497.4, 23.0, 179.5, 70.0245, 36.587]:
            self.do_exactly_equal_test(x, 0, 0)

    def testExactlyEqualFractions(self):
        # Test that equal Fraction values are exactly equal.
        F = Fraction
        for f in [F(1, 2), F(0), F(5, 3), F(9, 7), F(35, 36), F(3, 7)]:
            self.do_exactly_equal_test(f, 0, 0)

    def testExactlyEqualDecimals(self):
        # Test that equal Decimal values are exactly equal.
        D = Decimal
        for d in map(D, "8.2 31.274 912.04 16.745 1.2047".split()):
            self.do_exactly_equal_test(d, 0, 0)

    def testExactlyEqualAbsolute(self):
        # Test that equal values are exactly equal with an absolute error.
        for n in [16, 1013, 1372, 1198, 971, 4]:
            # Test as ints.
            self.do_exactly_equal_test(n, 0.01, 0)
            # Test as floats.
            self.do_exactly_equal_test(n/10, 0.01, 0)
            # Test as Fractions.
            f = Fraction(n, 1234)
            self.do_exactly_equal_test(f, 0.01, 0)

    def testExactlyEqualAbsoluteDecimals(self):
        # Test equal Decimal values are exactly equal with an absolute error.
        self.do_exactly_equal_test(Decimal("3.571"), Decimal("0.01"), 0)
        self.do_exactly_equal_test(-Decimal("81.3971"), Decimal("0.01"), 0)

    def testExactlyEqualRelative(self):
        # Test that equal values are exactly equal with a relative error.
        for x in [8347, 101.3, -7910.28, Fraction(5, 21)]:
            self.do_exactly_equal_test(x, 0, 0.01)
        self.do_exactly_equal_test(Decimal("11.68"), 0, Decimal("0.01"))

    def testExactlyEqualBoth(self):
        # Test that equal values are equal when both tol and rel are given.
        for x in [41017, 16.742, -813.02, Fraction(3, 8)]:
            self.do_exactly_equal_test(x, 0.1, 0.01)
        D = Decimal
        self.do_exactly_equal_test(D("7.2"), D("0.1"), D("0.01"))


class ApproxEqualUnequalTest(unittest.TestCase):
    # Unequal values should compare unequal with zero error tolerances.
    # Test cases for unequal values, with exact equality test.

    def do_exactly_unequal_test(self, x):
        for a in (x, -x):
            result = approx_equal(a, a+1, tol=0, rel=0)
            self.assertFalse(result, 'inequality failure for x=%r' % a)

    def testExactlyUnequalInts(self):
        # Test unequal int values are unequal with zero error tolerance.
        for n in [951, 572305, 478, 917, 17240]:
            self.do_exactly_unequal_test(n)

    def testExactlyUnequalFloats(self):
        # Test unequal float values are unequal with zero error tolerance.
        for x in [9.51, 5723.05, 47.8, 9.17, 17.24]:
            self.do_exactly_unequal_test(x)

    def testExactlyUnequalFractions(self):
        # Test that unequal Fractions are unequal with zero error tolerance.
        F = Fraction
        for f in [F(1, 5), F(7, 9), F(12, 11), F(101, 99023)]:
            self.do_exactly_unequal_test(f)

    def testExactlyUnequalDecimals(self):
        # Test that unequal Decimals are unequal with zero error tolerance.
        for d in map(Decimal, "3.1415 298.12 3.47 18.996 0.00245".split()):
            self.do_exactly_unequal_test(d)


class ApproxEqualInexactTest(unittest.TestCase):
    # Inexact test cases for approx_error.
    # Test cases when comparing two values that are not exactly equal.

    # === Absolute error tests ===

    def do_approx_equal_abs_test(self, x, delta):
        template = "Test failure for x={!r}, y={!r}"
        for y in (x + delta, x - delta):
            msg = template.format(x, y)
            self.assertTrue(approx_equal(x, y, tol=2*delta, rel=0), msg)
            self.assertFalse(approx_equal(x, y, tol=delta/2, rel=0), msg)

    def testApproxEqualAbsoluteInts(self):
        # Test approximate equality of ints with an absolute error.
        for n in [-10737, -1975, -7, -2, 0, 1, 9, 37, 423, 9874, 23789110]:
            self.do_approx_equal_abs_test(n, 10)
            self.do_approx_equal_abs_test(n, 2)

    def testApproxEqualAbsoluteFloats(self):
        # Test approximate equality of floats with an absolute error.
        for x in [-284.126, -97.1, -3.4, -2.15, 0.5, 1.0, 7.8, 4.23, 3817.4]:
            self.do_approx_equal_abs_test(x, 1.5)
            self.do_approx_equal_abs_test(x, 0.01)
            self.do_approx_equal_abs_test(x, 0.0001)

    def testApproxEqualAbsoluteFractions(self):
        # Test approximate equality of Fractions with an absolute error.
        delta = Fraction(1, 29)
        numerators = [-84, -15, -2, -1, 0, 1, 5, 17, 23, 34, 71]
        for f in (Fraction(n, 29) for n in numerators):
            self.do_approx_equal_abs_test(f, delta)
            self.do_approx_equal_abs_test(f, float(delta))

    def testApproxEqualAbsoluteDecimals(self):
        # Test approximate equality of Decimals with an absolute error.
        delta = Decimal("0.01")
        for d in map(Decimal, "1.0 3.5 36.08 61.79 7912.3648".split()):
            self.do_approx_equal_abs_test(d, delta)
            self.do_approx_equal_abs_test(-d, delta)

    def testCrossZero(self):
        # Test for the case of the two values having opposite signs.
        self.assertTrue(approx_equal(1e-5, -1e-5, tol=1e-4, rel=0))

    # === Relative error tests ===

    def do_approx_equal_rel_test(self, x, delta):
        template = "Test failure for x={!r}, y={!r}"
        for y in (x*(1+delta), x*(1-delta)):
            msg = template.format(x, y)
            self.assertTrue(approx_equal(x, y, tol=0, rel=2*delta), msg)
            self.assertFalse(approx_equal(x, y, tol=0, rel=delta/2), msg)

    def testApproxEqualRelativeInts(self):
        # Test approximate equality of ints with a relative error.
        self.assertTrue(approx_equal(64, 47, tol=0, rel=0.36))
        self.assertTrue(approx_equal(64, 47, tol=0, rel=0.37))
        # ---
        self.assertTrue(approx_equal(449, 512, tol=0, rel=0.125))
        self.assertTrue(approx_equal(448, 512, tol=0, rel=0.125))
        self.assertFalse(approx_equal(447, 512, tol=0, rel=0.125))

    def testApproxEqualRelativeFloats(self):
        # Test approximate equality of floats with a relative error.
        for x in [-178.34, -0.1, 0.1, 1.0, 36.97, 2847.136, 9145.074]:
            self.do_approx_equal_rel_test(x, 0.02)
            self.do_approx_equal_rel_test(x, 0.0001)

    def testApproxEqualRelativeFractions(self):
        # Test approximate equality of Fractions with a relative error.
        F = Fraction
        delta = Fraction(3, 8)
        for f in [F(3, 84), F(17, 30), F(49, 50), F(92, 85)]:
            for d in (delta, float(delta)):
                self.do_approx_equal_rel_test(f, d)
                self.do_approx_equal_rel_test(-f, d)

    def testApproxEqualRelativeDecimals(self):
        # Test approximate equality of Decimals with a relative error.
        for d in map(Decimal, "0.02 1.0 5.7 13.67 94.138 91027.9321".split()):
            self.do_approx_equal_rel_test(d, Decimal("0.001"))
            self.do_approx_equal_rel_test(-d, Decimal("0.05"))

    # === Both absolute and relative error tests ===

    # There are four cases to consider:
    #   1) actual error <= both absolute and relative error
    #   2) actual error <= absolute error but > relative error
    #   3) actual error <= relative error but > absolute error
    #   4) actual error > both absolute and relative error

    def do_check_both(self, a, b, tol, rel, tol_flag, rel_flag):
        check = self.assertTrue if tol_flag else self.assertFalse
        check(approx_equal(a, b, tol=tol, rel=0))
        check = self.assertTrue if rel_flag else self.assertFalse
        check(approx_equal(a, b, tol=0, rel=rel))
        check = self.assertTrue if (tol_flag or rel_flag) else self.assertFalse
        check(approx_equal(a, b, tol=tol, rel=rel))

    def testApproxEqualBoth1(self):
        # Test actual error <= both absolute and relative error.
        self.do_check_both(7.955, 7.952, 0.004, 3.8e-4, True, True)
        self.do_check_both(-7.387, -7.386, 0.002, 0.0002, True, True)

    def testApproxEqualBoth2(self):
        # Test actual error <= absolute error but > relative error.
        self.do_check_both(7.955, 7.952, 0.004, 3.7e-4, True, False)

    def testApproxEqualBoth3(self):
        # Test actual error <= relative error but > absolute error.
        self.do_check_both(7.955, 7.952, 0.001, 3.8e-4, False, True)

    def testApproxEqualBoth4(self):
        # Test actual error > both absolute and relative error.
        self.do_check_both(2.78, 2.75, 0.01, 0.001, False, False)
        self.do_check_both(971.44, 971.47, 0.02, 3e-5, False, False)


class ApproxEqualSpecialsTest(unittest.TestCase):
    # Test approx_equal with NANs and INFs and zeroes.

    def testInf(self):
        for type_ in (float, Decimal):
            inf = type_('inf')
            self.assertTrue(approx_equal(inf, inf))
            self.assertTrue(approx_equal(inf, inf, 0, 0))
            self.assertTrue(approx_equal(inf, inf, 1, 0.01))
            self.assertTrue(approx_equal(-inf, -inf))
            self.assertFalse(approx_equal(inf, -inf))
            self.assertFalse(approx_equal(inf, 1000))

    def testNan(self):
        for type_ in (float, Decimal):
            nan = type_('nan')
            for other in (nan, type_('inf'), 1000):
                self.assertFalse(approx_equal(nan, other))

    def testFloatZeroes(self):
        nzero = math.copysign(0.0, -1)
        self.assertTrue(approx_equal(nzero, 0.0, tol=0.1, rel=0.1))

    def testDecimalZeroes(self):
        nzero = Decimal("-0.0")
        self.assertTrue(approx_equal(nzero, Decimal(0), tol=0.1, rel=0.1))


class TestApproxEqualErrors(unittest.TestCase):
    # Test error conditions of approx_equal.

    def testBadTol(self):
        # Test negative tol raises.
        self.assertRaises(ValueError, approx_equal, 100, 100, -1, 0.1)

    def testBadRel(self):
        # Test negative rel raises.
        self.assertRaises(ValueError, approx_equal, 100, 100, 1, -0.1)


# === Tests for NumericTestCase ===

# The formatting routine that generates the error messages is complex enough
# that it too needs testing.

class TestNumericTestCase(unittest.TestCase):
    # The exact wording of NumericTestCase error messages is *not* guaranteed,
    # but we need to give them some sort of test to ensure that they are
    # generated correctly. As a compromise, we look for specific substrings
    # that are expected to be found even if the overall error message changes.

    def do_test(self, args):
        actual_msg = NumericTestCase._make_std_err_msg(*args)
        expected = self.generate_substrings(*args)
        for substring in expected:
            self.assertIn(substring, actual_msg)

    def testNumericTestCaseIsTestCase(self):
        # Ensure that NumericTestCase actually is a TestCase.
        self.assertTrue(issubclass(NumericTestCase, unittest.TestCase))

    def testErrorMsgNumeric(self):
        # Test the error message generated for numeric comparisons.
        args = (2.5, 4.0, 0.5, 0.25, None)
        self.do_test(args)

    def testErrorMsgSequence(self):
        # Test the error message generated for sequence comparisons.
        args = (3.75, 8.25, 1.25, 0.5, 7)
        self.do_test(args)

    def generate_substrings(self, actual, expected, tol, rel, idx):
        """Return substrings we expect to see in error messages."""
        abs_err, rel_err = _calc_errors(actual, expected)
        substrings = [
                'actual value %r' % actual,
                'expected %r' % expected,
                'tol=%r' % tol,
                'rel=%r' % rel,
                'absolute error = %r' % abs_err,
                'relative error = %r' % rel_err,
                ]
        if idx is not None:
            substrings.append('differ at index %d' % idx)
        return substrings


# === Self-testing ===

def load_tests(loader, tests, ignore):
    """Used for doctest/unittest integration."""
    tests.addTests(doctest.DocTestSuite())
    return tests


if __name__ == '__main__':
    """Run module unit tests and doc tests."""
    unittest.main()


