#! /usr/bin/env python

"""
test.py

Written by Geremy Condra
Licensed under the Python License
Released 21 October 2010

This module contains some lightweight tests for the stats module.
"""

import unittest
from math import pi, e

class UsabilityTests(unittest.TestCase):
	
    def setUp(self):
        self.sequence = list(range(1000))
        self.vanishing_sequence = [.1/float(i) for i in self.sequence]
        self.pi_multiples = [pi*i for i in self.sequence]
        self.e_multiples = [e*i for i in self.sequence]
        self.inv_pi_multiples = [pi*i for i in self.vanishing_sequence]
        self.inv_e_multiples = [e*i for i in self.vanishing_sequence]

    def test_mean(self):
        pass

    def test_harmonic_mean(self):
        pass

    def test_geometric_mean(self):
        pass

    def test_quadratic_mean(self):
        pass

    def test_median(self):
        pass

    def test_mode(self):
        pass

    def test_midrange(self):
        pass

    def test_stdev(self):
        pass

    def test_pstdev(self):
        pass

    def test_variance(self):
        pass

    def test_pvariance(self):
        pass

    def test_range(self):
        pass

    def test_iqr(self):
        pass

    def test_average_deviation(self):
        pass

    def test_corr(self):
        pass

    def test_cov(self):
        pass

    def test_pcov(self):
        pass

    def test_errsumsq(self):
        pass

    def test_linr(self):
        pass

    def test_sum(self):
        pass

    def test_sumsq(self):
        pass

    def test_product(self):
        pass

    def test_xsums(self):
        pass

    def test_xysums(self):
        pass

    def test_Sxx(self):
        pass

    def test_Syy(self):
        pass

    def test_Sxy(self):
        pass

    def test_sterrmean(self):
        pass


class PrecisionTests(unittest.TestCase):
    pass

if __name__ == "__main__":
    unittest.main()
