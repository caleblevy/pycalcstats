#!/usr/bin/env python3

##  Copyright (c) 2011 Steven D'Aprano.
##  See the file stats/__init__.py for the licence terms for this software.

"""
Test suite for the stats.order module.

"""

from stats.tests import NumericTestCase
import stats.order


# Test order statistics
# ---------------------

class DrMathTests(NumericTestCase):
    # Sample data for testing quartiles taken from Dr Math page:
    # http://mathforum.org/library/drmath/view/60969.html
    # Q2 values are not checked in this test.
    A = range(1, 9)
    B = range(1, 10)
    C = range(1, 11)
    D = range(1, 12)

    def testInclusive(self):
        f = stats._Quartiles.inclusive
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
        f = stats._Quartiles.exclusive
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
        f = stats._Quartiles.ms
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
        f = stats._Quartiles.minitab
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
        f = stats._Quartiles.excel
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


class QuartileAliasesTest(NumericTestCase):
    allowed_methods = set(stats._Quartiles.QUARTILE_MAP.keys())

    def testAliasesMapping(self):
        # Test that the quartile function exposes a mapping of aliases.
        self.assertTrue(hasattr(stats.quartiles, 'aliases'))
        aliases = stats.quartiles.aliases
        self.assertTrue(isinstance(aliases, collections.Mapping))
        self.assertTrue(aliases)

    def testAliasesValues(self):
        for method in stats.quartiles.aliases.values():
            self.assertTrue(method in self.allowed_methods)


class QuartileTest(NumericTestCase):
    # Schemes to be tested.
    schemes = [1, 2, 3, 4, 5, 6]

    def __init__(self, *args, **kwargs):
        NumericTestCase.__init__(self, *args, **kwargs)
        self.func = stats.quartiles

    # Helper methods:

    def compare_sorted_with_unsorted(self, n, scheme):
        data = list(range(n))
        result1 = self.func(data, scheme)
        random.shuffle(data)
        result2 = self.func(data, scheme)
        self.assertEqual(result1, result2)

    def compare_types(self, n, scheme):
        data = range(n)
        result1 = self.func(data, scheme)
        data = list(data)
        result2 = self.func(data, scheme)
        result3 = self.func(tuple(data), scheme)
        result4 = self.func(iter(data), scheme)
        self.assertEqual(result1, result2)
        self.assertEqual(result1, result3)
        self.assertEqual(result1, result4)

    def expect_failure(self, data):
        self.assertRaises(ValueError, self.func, data)
        for scheme in self.schemes:
            self.assertRaises(ValueError, self.func, data, scheme)

    # Generic tests that don't care about the specific values:

    def testNoSorting(self):
        """Test that quartiles doesn't sort in place."""
        data = [2, 4, 1, 3, 0, 5]
        save = data[:]
        assert save is not data
        assert data != sorted(data)
        for scheme in self.schemes:
            _ = self.func(data, scheme)
            self.assertEqual(data, save)

    def testTooFewItems(self):
        for data in ([], [1], [1, 2]):
            self.expect_failure(data)

    def testSorted(self):
        # Test that sorted and unsorted data give the same results.
        for n in (8, 9, 10, 11):  # n%4 -> 0...3
            for scheme in self.schemes:
                self.compare_sorted_with_unsorted(n, scheme)

    def testIter(self):
        # Test that iterators and sequences give the same result.
        for n in (8, 9, 10, 11):  # n%4 -> 0...3
            for scheme in self.schemes:
                self.compare_types(n, scheme)

    def testBadScheme(self):
        data = range(20)
        for scheme in ('', 'spam', 1.5, -1.5, -2):
            self.assertRaises(ValueError, self.func, data, scheme)

    def testCaseInsensitive(self):
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
        save_scheme = stats.QUARTILE_DEFAULT
        schemes = [1, 2, 3, 4, 5, 6, 'excel', 'minitab']
        try:
            for scheme in schemes:
                stats.QUARTILE_DEFAULT = scheme
                a = stats.quartiles(data)
                b = stats.quartiles(data, scheme)
                self.assertEqual(a, b)
        finally:
            stats.QUARTILE_DEFAULT = save_scheme

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
        self.assertEqual((0.5, 1, 1.5), f(range(3), scheme))
        self.assertEqual((0.75, 1.5, 2.25), f(range(4), scheme))
        self.assertEqual((1, 2, 3), f(range(5), scheme))
        self.assertEqual((1.25, 2.5, 3.75), f(range(6), scheme))
        self.assertEqual((1.5, 3, 4.5), f(range(7), scheme))
        self.assertEqual((1.75, 3.5, 5.25), f(range(8), scheme))
        self.assertEqual((2, 4, 6), f(range(9), scheme))
        self.assertEqual((2.25, 4.5, 6.75), f(range(10), scheme))
        self.assertEqual((2.5, 5, 7.5), f(range(11), scheme))
        self.assertEqual((2.75, 5.5, 8.25), f(range(12), scheme))
        self.assertEqual((3, 6, 9), f(range(13), scheme))
        self.assertEqual((3.25, 6.5, 9.75), f(range(14), scheme))
        self.assertEqual((3.5, 7, 10.5), f(range(15), scheme))

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


class HingesTest(NumericTestCase):
    def __init__(self, *args, **kwargs):
        NumericTestCase.__init__(self, *args, **kwargs)
        self.func = stats.hinges

    def testNoSorting(self):
        # hinges() does not sort in place.
        data = [2, 4, 1, 3, 0, 5]
        save = data[:]
        assert save is not data
        assert data != sorted(data)
        _ = self.func(data)
        self.assertEqual(data, save)

    def testTooFewItems(self):
        for data in ([], [1], [1, 2]):
            self.assertRaises(ValueError, self.func, data)

    def testSorted(self):
        # Test that sorted and unsorted data give the same results.
        for n in (40, 41, 42, 43):  # n%4 -> 0...3
            data = list(range(n))
            result1 = self.func(data)
            random.shuffle(data)
            result2 = self.func(data)
            self.assertEqual(result1, result2)

    def testTypes(self):
        # Test that iterators and sequences give the same result.
        for n in (40, 41, 42, 43):
            data = range(n)
            result1 = self.func(data)
            data = list(data)
            result2 = self.func(data)
            result3 = self.func(tuple(data))
            result4 = self.func(iter(data))
            self.assertEqual(result1, result2)
            self.assertEqual(result1, result3)
            self.assertEqual(result1, result4)

    def testHinges(self):
        f = self.func
        for n in range(3, 25):
            data = range(n)
            self.assertEqual(f(data), stats.quartiles(data, 'hinges'))


class QuantileBehaviourTest(NumericTestCase):
    # Test behaviour of quantile function without caring about
    # the actual values returned.

    def __init__(self, *args, **kwargs):
        NumericTestCase.__init__(self, *args, **kwargs)
        self.func = stats.quantile

    def testSorting(self):
        # Test that quantile doesn't sort in place.
        data = [2, 4, 1, 3, 0, 5]
        assert data != sorted(data)
        save = data[:]
        assert save is not data
        _ = self.func(data, 0.9)
        self.assertEqual(data, save)

    def testQuantileArgOutOfRange(self):
        data = [1, 2, 3, 4]
        self.assertRaises(ValueError, self.func, data, -0.1)
        self.assertRaises(ValueError, self.func, data, 1.1)

    def testTooFewItems(self):
        self.assertRaises(ValueError, self.func, [], 0.1)
        self.assertRaises(ValueError, self.func, [1], 0.1)

    def testDefaultScheme(self):
        data = list(range(51))
        random.shuffle(data)
        assert len(data) % 4 == 3
        save_scheme = stats.QUANTILE_DEFAULT
        schemes = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            'excel', 'minitab',
            (0.375, 0.25, 0, 1),
            ]
        try:
            for scheme in schemes:
                p = random.random()
                stats.QUANTILE_DEFAULT = scheme
                a = stats.quantile(data, p)
                b = stats.quantile(data, p, scheme)
                self.assertEqual(a, b)
        finally:
            stats.QUANTILE_DEFAULT = save_scheme


class QuantileValueTest(NumericTestCase):
    # Tests of quantile function where we do care about the actual
    # values returned.

    def __init__(self, *args, **kwargs):
        NumericTestCase.__init__(self, *args, **kwargs)
        self.func = stats.quantile

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
            result = stats.quantile(data, p, scheme=10)
            self.assertApproxEqual(expected[i], result, tol=1e-12, rel=None)


class QuantilesCompareWithR(NumericTestCase):
    # Compare results of calling quantile() against results from R.
    tol = 1e-3
    rel = None

    def __init__(self, *args, **kwargs):
        NumericTestCase.__init__(self, *args, **kwargs)
        self.read_data('quantiles.dat')

    def read_data(self, filename):
        # Read data from external test data file generated using R.
        expected = {}
        with open(filename, 'r') as data:
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
        a = [stats.quantile(self.data, p, scheme=scheme) for p in fractiles]
        b = self.expected[scheme]
        for x,y in zip(a, b):
            self.assertApproxEqual(x, y)

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
    data1 = list(range(1000, 2000, 100))
    data2 = list(range(2000, 3001, 100))
    assert len(data1)%2 == 0
    assert len(data2)%2 == 1

    fractions = [0.0, 0.01, 0.1, 0.2, 0.25, 0.31, 0.42, 0.5, 0.55,
                0.62, 0.75, 0.83, 0.9, 0.95, 0.99, 1.0]

    def compareMethods(self, scheme, params):
        for p in self.fractions:
            a = stats.quantile(self.data1, p, scheme=scheme)
            b = stats.quantile(self.data1, p, scheme=params)
            self.assertEqual(a, b, "%s != %s; p=%f" % (a, b, p))
            a = stats.quantile(self.data2, p, scheme=scheme)
            b = stats.quantile(self.data2, p, scheme=params)
            self.assertEqual(a, b, "%s != %s; p=%f" % (a, b, p))

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


class DecileTest(NumericTestCase):
    def testSimple(self):
        data = range(1, 11)
        for i in range(1, 11):
            self.assertEqual(stats.decile(data, i, scheme=1), i)


class PercentileTest(NumericTestCase):
    def testSimple(self):
        data = range(1, 101)
        for i in range(1, 101):
            self.assertEqual(stats.percentile(data, i, scheme=1), i)


class IQRTest(NumericTestCase):

    def testBadScheme(self):
        # Test that a bad scheme raises an exception.
        for scheme in (-1, 1.5, "spam"):
            self.assertRaises(ValueError, stats.iqr, [1, 2, 3, 4], scheme)

    def testFailure(self):
        # Test that too few items raises an exception.
        for data in ([], [1], [2, 3]):
            self.assertRaises(ValueError, stats.iqr, data)

    def testTriplet(self):
        # Test that data consisting of three items behaves as expected.
        data = [1, 5, 12]
        self.assertEqual(stats.iqr(data, 'inclusive'), 5.5)
        self.assertEqual(stats.iqr(data, 'exclusive'), 11)

    def testCaseInsensitive(self):
        data = [1, 2, 3, 6, 9, 12, 18, 22]
        for name, num in stats.quartiles.aliases.items():
            a = stats.iqr(data, name.lower())
            b = stats.iqr(data, name.upper())
            c = stats.iqr(data, name.title())
            d = stats.iqr(data, num)
            self.assertEqual(a, b)
            self.assertEqual(a, c)
            self.assertEqual(a, d)

    def testDefaultScheme(self):
        # Test that iqr inherits the global default for quartiles.
        data = list(range(51))
        random.shuffle(data)
        assert len(data) % 4 == 3
        save_scheme = stats.QUARTILE_DEFAULT
        schemes = [1, 2, 3, 4, 5, 6]
        try:
            for scheme in schemes:
                stats.QUARTILE_DEFAULT = scheme
                a = stats.iqr(data)
                b = stats.iqr(data, scheme)
                self.assertEqual(a, b)
        finally:
            stats.QUARTILE_DEFAULT = save_scheme

    def same_result(self, data, scheme):
        # Check that data gives the same result, no matter what order
        # it is given in.
        assert len(data) > 2
        if len(data) <= 7:
            # Exhaustively try every permutation for small amounts of data.
            perms = itertools.permutations(data)
        else:
            # Take a random sample for larger amounts of data.
            data = list(data)
            perms = []
            for _ in range(50):
                random.shuffle(data)
                perms.append(data[:])
        results = [stats.iqr(perm, scheme) for perm in perms]
        assert len(results) > 1
        self.assertTrue(len(set(results)) == 1)

    def testCompareOrder(self):
        # Ensure results don't depend on the order of the input.
        for scheme in [1, 2, 3, 4, 5, 6]:
            for size in range(3, 12):  # size % 4 -> 3,0,1,2 ...
                self.same_result(range(size), scheme)


