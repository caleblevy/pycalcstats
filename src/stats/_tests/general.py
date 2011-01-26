#!/usr/bin/env python3

##  Copyright (c) 2011 Steven D'Aprano.
##  See the file stats/__init__.py for the licence terms for this software.

"""
Miscellaneous and general tests that don't fit into any other of the specific
package modules.

"""

import math
import os
import pickle
import zipfile

from stats._tests import NumericTestCase

import stats
import stats.multivar
import stats.order
import stats.univar



class CompareAgainstNumpyResultsTest(NumericTestCase):
    # Test the results we generate against some numpy equivalents.
    places = 8

    def __init__(self, *args, **kwargs):
        NumericTestCase.__init__(self, *args, **kwargs)
        # Read data from external test data file.
        # (In this case, produced by numpy and Python 2.5.)
        location = self.get_data_location('support/test_data.zip')
        # Now read the data from that file.
        zf = zipfile.ZipFile(location, 'r')
        self.data = pickle.loads(zf.read('data.pkl'))
        self.expected = pickle.loads(zf.read('results.pkl'))
        zf.close()

    def get_data_location(self, filename):
        # First we have to find our base location.
        import stats._tests
        location = os.path.split(stats._tests.__file__)[0]
        # Now add the filename to it.
        return os.path.join(location, filename)

    # FIXME assertAlmostEqual is not really the right way to do these
    # tests, as decimal places != significant figures.
    def testSum(self):
        result = stats.sum(self.data)
        expected = self.expected['sum']
        n = int(math.log(result, 10))  # Yuck.
        self.assertAlmostEqual(result, expected, places=self.places-n)

    def testProduct(self):
        result = stats.univar.product(self.data)
        expected = self.expected['product']
        self.assertAlmostEqual(result, expected, places=self.places)

    def testMean(self):
        result = stats.mean(self.data)
        expected = self.expected['mean']
        self.assertAlmostEqual(result, expected, places=self.places)

    def testRange(self):
        result = stats.order.range(self.data)
        expected = self.expected['range']
        self.assertAlmostEqual(result, expected, places=self.places)

    def testMidrange(self):
        result = stats.order.midrange(self.data)
        expected = self.expected['midrange']
        self.assertAlmostEqual(result, expected, places=self.places)

    def testPStdev(self):
        result = stats.pstdev(self.data)
        expected = self.expected['pstdev']
        self.assertAlmostEqual(result, expected, places=self.places)

    def testPVar(self):
        result = stats.pvariance(self.data)
        expected = self.expected['pvariance']
        self.assertAlmostEqual(result, expected, places=self.places)


class AssortedResultsTest(NumericTestCase):
    # Test some assorted statistical results against exact results
    # calculated by hand, and confirmed by HP-48GX calculations.
    places = 16

    def __init__(self, *args, **kwargs):
        NumericTestCase.__init__(self, *args, **kwargs)
        self.xdata = [1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 3/2, 5/2,
                      7/2, 9/2, 11/2, 13/2, 15/2, 17/2, 19/2]
        self.ydata = [1/4, 1/2, 3/2, 1, 1/2, 3/2, 1, 5/4, 5/2, 7/4,
                      9/4, 11/4, 11/4, 7/4, 13/4, 17/4]
        assert len(self.xdata) == len(self.ydata) == 16

    def testSums(self):
        Sx = stats.sum(self.xdata)
        Sy = stats.sum(self.ydata)
        self.assertAlmostEqual(Sx, 3295/64, places=self.places)
        self.assertAlmostEqual(Sy, 115/4, places=self.places)

    def testSumSqs(self):
        Sx2 = stats.sum(x**2 for x in self.xdata)
        Sy2 = stats.sum(x**2 for x in self.ydata)
        self.assertAlmostEqual(Sx2, 1366357/4096, places=self.places)
        self.assertAlmostEqual(Sy2, 1117/16, places=self.places)

    def testMeans(self):
        x = stats.mean(self.xdata)
        y = stats.mean(self.ydata)
        self.assertAlmostEqual(x, 3295/1024, places=self.places)
        self.assertAlmostEqual(y, 115/64, places=self.places)

    def testOtherSums(self):
        Sxx = stats.multivar.Sxx(zip(self.xdata, self.ydata))
        Syy = stats.multivar.Syy(zip(self.xdata, self.ydata))
        Sxy = stats.multivar.Sxy(zip(self.xdata, self.ydata))
        self.assertAlmostEqual(Sxx, 11004687/4096, places=self.places)
        self.assertAlmostEqual(Syy, 4647/16, places=self.places)
        self.assertAlmostEqual(Sxy, 197027/256, places=self.places)

    def testPVar(self):
        sx2 = stats.pvariance(self.xdata)
        sy2 = stats.pvariance(self.ydata)
        self.assertAlmostEqual(sx2, 11004687/1048576, places=self.places)
        self.assertAlmostEqual(sy2, 4647/4096, places=self.places)

    def testVar(self):
        sx2 = stats.variance(self.xdata)
        sy2 = stats.variance(self.ydata)
        self.assertAlmostEqual(sx2, 11004687/983040, places=self.places)
        self.assertAlmostEqual(sy2, 4647/3840, places=self.places)

    def testPCov(self):
        v = stats.multivar.pcov(self.xdata, self.ydata)
        self.assertAlmostEqual(v, 197027/65536, places=self.places)

    def testCov(self):
        v = stats.multivar.cov(self.xdata, self.ydata)
        self.assertAlmostEqual(v, 197027/61440, places=self.places)

    def testErrSumSq(self):
        se = stats.multivar.errsumsq(self.xdata, self.ydata)
        self.assertAlmostEqual(se, 96243295/308131236, places=self.places)

    def testLinr(self):
        a, b = stats.multivar.linr(self.xdata, self.ydata)
        expected_b = 3152432/11004687
        expected_a = 115/64 - expected_b*3295/1024
        self.assertAlmostEqual(a, expected_a, places=self.places)
        self.assertAlmostEqual(b, expected_b, places=self.places)

    def testCorr(self):
        r = stats.multivar.corr(zip(self.xdata, self.ydata))
        Sxx = 11004687/4096
        Syy = 4647/16
        Sxy = 197027/256
        expected = Sxy/math.sqrt(Sxx*Syy)
        self.assertAlmostEqual(r, expected, places=15)

