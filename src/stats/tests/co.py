#!/usr/bin/env python3

##  Copyright (c) 2011 Steven D'Aprano.
##  See the file stats/__init__.py for the licence terms for this software.

"""
Test suite for the stats.co module.

"""

from stats.tests import NumericTestCase
import stats.tests.common as common

# The module to be tested:
import stats.co


"""
class PVariance1Test(PVarianceTest):
    func = stats.pvariance1


class Variance1Test(VarianceTest):
    func = stats.variance1

    def testWithLargeData(self):
        small_data = [random.gauss(3.5, 2.5) for _ in range(1000)]
        a = stats.variance(small_data)
        # We expect a to be close to the exact result for the variance,
        # namely 5.5**2, but if it's not, that's just a fluke of the random
        # sample. Either way, it doesn't matter.
        b = self.func(small_data)
        self.assertApproxEqual(a, b, tol=1e-12)

        def big_data():
            for _ in range(100):
                for x in small_data:
                    yield x

        c = self.func(big_data())
        # Expect variance to be unchanged, except for rounding errors.
        self.assertApproxEqual(a, c, tol=None, rel=1e-3)

class PStdev1Test(PStdevTest):
    func = stats.pstdev1


class Stdev1Test(StdevTest):
    func = stats.stdev1


class WeightedRunningAverageTest(RunningAverageTest):
    def __init__(self, *args, **kwargs):
        RunningAverageTest.__init__(self, *args, **kwargs)
        self.func = stats.weighted_running_average

    def testFinal(self):
        # Test the final result has the expected value.
        data = [64, 32, 16, 8, 4, 2, 1]
        results = list(self.func(data))
        self.assertEqual(results[-1], 4)


"""