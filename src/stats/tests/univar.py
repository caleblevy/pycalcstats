#!/usr/bin/env python3

##  Copyright (c) 2011 Steven D'Aprano.
##  See the file stats/__init__.py for the licence terms for this software.

"""
Test suite for the stats.univar module.

"""

from stats.tests import NumericTestCase
import stats.univar


class HarmonicMeanTest(MeanTest):
    func = stats.harmonic_mean
    expected = 3.4995090404755
    rel = 1e-8


class GeometricMeanTest(MeanTest):
    func = stats.geometric_mean
    expected = 4.56188290183
    rel = 1e-11

    @unittest.skip('geometric mean currently too inaccurate')
    def testBigData(self):
        super().testBigData()

    def testNegative(self):
        data = [1.0, 2.0, -3.0, 4.0]
        assert any(x < 0.0 for x in data)
        self.assertRaises(ValueError, self.func, data)

    def testZero(self):
        data = [1.0, 2.0, 0.0, 4.0]
        assert any(x == 0.0 for x in data)
        self.assertEqual(self.func(data), 0.0)


class QuadraticMeanTest(MeanTest):
    func = stats.quadratic_mean
    expected = 6.19004577259
    rel = 1e-8

    def testNegative(self):
        data = [-x for x in self.data]
        self.assertApproxEqual(self.func(data), self.expected)


# Test statistics of circular quantities
# --------------------------------------

class CircularMeanTest(NumericTestCase):
    tol = 1e-12

    def testDefaultDegrees(self):
        # Test that degrees are the default.
        data = [355, 5, 15, 320, 45]
        theta = stats.circular_mean(data)
        phi = stats.circular_mean(data, True)
        assert stats.circular_mean(data, False) != theta
        self.assertEqual(theta, phi)

    def testRadians(self):
        # Test that degrees and radians (usually) give different results.
        data = [355, 5, 15, 320, 45]
        a = stats.circular_mean(data, True)
        b = stats.circular_mean(data, False)
        self.assertNotEquals(a, b)

    def testEmpty(self):
        self.assertRaises(ValueError, stats.circular_mean, [])

    def testSingleton(self):
        for x in (-1.0, 0.0, 1.0, 3.0):
            self.assertEqual(stats.circular_mean([x], False), x)
            self.assertApproxEqual(stats.circular_mean([x], True), x)

    def testNegatives(self):
        data1 = [355, 5, 15, 320, 45]
        theta = stats.circular_mean(data1)
        data2 = [d-360 if d > 180 else d for d in data1]
        phi = stats.circular_mean(data2)
        self.assertApproxEqual(theta, phi)

    def testIter(self):
        theta = stats.circular_mean(iter([355, 5, 15]))
        self.assertApproxEqual(theta, 5.0)

    def testSmall(self):
        t = stats.circular_mean([0, 360])
        self.assertApproxEqual(t, 0.0)
        t = stats.circular_mean([10, 20, 30])
        self.assertApproxEqual(t, 20.0)
        t = stats.circular_mean([355, 5, 15])
        self.assertApproxEqual(t, 5.0)

    def testFullCircle(self):
        # Test with angle > full circle.
        theta = stats.circular_mean([3, 363])
        self.assertApproxEqual(theta, 3)

    def testBig(self):
        pi = math.pi
        # Generate angles between pi/2 and 3*pi/2, with expected mean of pi.
        delta = pi/1000
        data = [pi/2 + i*delta for i in range(1000)]
        data.append(3*pi/2)
        assert data[0] == pi/2
        assert len(data) == 1001
        random.shuffle(data)
        theta = stats.circular_mean(data, False)
        self.assertApproxEqual(theta, pi)
        # Now try the same with angles in the first and fourth quadrants.
        data = [0.0]
        for i in range(1, 501):
            data.append(i*delta)
            data.append(2*pi - i*delta)
        assert len(data) == 1001
        random.shuffle(data)
        theta = stats.circular_mean(data, False)
        self.assertApproxEqual(theta, 0.0)


