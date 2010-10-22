#! /usr/bin/env python3

"""
test.py

Written by Geremy Condra
Licensed under the Python License
Released 21 October 2010

This module contains some lightweight tests for the stats module.
"""

import unittest
from math import pi, e

import stats

#------------------------------------------------------------------------------#
#                    Arithmetically Inconvenient Types                         #
#------------------------------------------------------------------------------#

class NoAddition(int):

    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self

    def __div__(self, other):
        return self

    def __rdiv__(self, other):
        return self

    def __add__(self, other):
        raise NotImplementedError("No addition!")

    def __radd__(self, other):
        raise NotImplementedError("No addition!")


class NoMultiplication(int):

    def __add__(self, other):
        return self

    def __radd__(self, other):
        return self

    def __div__(self, other):
        return self

    def __rdiv__(self, other):
        return self

    def __mul__(self, other):
        raise NotImplementedError("No multiplication!")

    def __rmul__(self, other):
        raise NotImplementedError("No multiplication!")


class NoDivision(int):

    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self

    def __add__(self, other):
        return self

    def __radd__(self, other):
        return self

    def __div__(self, other):
        raise NotImplementedError("No division!")

    def __rdiv__(self, other):
        raise NotImplementedError("No division!")

#------------------------------------------------------------------------------#
#                           Test Sequences                                     #
#------------------------------------------------------------------------------#

# a few list sequences
sequence = list(range(1, 1001))
vanishing_sequence = [.1/float(i) for i in sequence]
pi_multiples = [pi*i for i in sequence]
e_multiples = [e*i for i in sequence]
inv_pi_multiples = [pi*i for i in vanishing_sequence]
inv_e_multiples = [e*i for i in vanishing_sequence]
zeros = [0] * 1000
ones = [1] * 1000
noadd = [NoAddition()] * 1000
nomul = [NoMultiplication()] * 1000
nodiv = [NoDivision()] * 1000
precision_sequence = [1, 1e100, 1, -1e100] * 10000
semiprecision_sequence = [0.123456789012345] * 10000

# the sequences that are only integers
int_seqs = [sequence, zeros, ones]
# the sequences that are only floats
float_seqs = [  vanishing_sequence,
                pi_multiples, e_multiples,
                inv_pi_multiples, 
                inv_e_multiples, 
                precision_sequence, 
                semiprecision_sequence
             ]

# the sequences that test various arithmetic properties
division_seqs = [zeros, nodiv]
addition_seqs = [noadd]
multiplication_seqs = [nomul]

# the sequences that test numeric precision
numeric_seqs = [precision_sequence, semiprecision_sequence]

# all of the above
seqs = [    sequence,
            vanishing_sequence,
            pi_multiples,
            e_multiples,
            inv_pi_multiples,
            inv_e_multiples,
            zeros,
            ones,
            noadd,
            nomul,
            nodiv,
            precision_sequence,
            semiprecision_sequence
        ]
#------------------------------------------------------------------------------#
#                                  Tests                                       #
#------------------------------------------------------------------------------#

class UsabilityTests(unittest.TestCase):

    def run_with_datasets(self, f, datasets):
        for dataset in datasets:
            f(dataset)

    def run_all(self, f, f_filter):
        # test to make sure it handles everthing except those filtered
        list_seqs = list(filter(f_filter, seqs))
        tuple_seqs = [tuple(i) for i in list_seqs]
        set_seqs = [set(i) for i in list_seqs]
        my_seqs = list_seqs + tuple_seqs + set_seqs
        self.run_with_datasets(f, my_seqs)

    def test_mean(self):
        # test to make sure it handles everthing except the noadd
        self.run_all(stats.mean, lambda x: x != noadd)

    def test_harmonic_mean(self):
        # test to make sure it handles everthing except the noadd
        self.run_all(stats.harmonic_mean, lambda x: x != noadd)

    def test_geometric_mean(self):
        # test to make sure it handles everthing except the noadd and nomul
        self.run_all(stats.geometric_mean, lambda x: x not in [noadd, nomul])

    def test_quadratic_mean(self):
        # test to make sure it handles everthing except the nomul
        self.run_all(stats.quadratic_mean, lambda x: x != nomul)

    def test_median(self):
        # test to make sure it handles everthing
        self.run_all(stats.median, lambda x: x not in [noadd])

    def test_mode(self):
        # test to make sure it handles everthing
        self.run_all(stats.mode, lambda x: x not in [noadd])

    def test_midrange(self):
        # test to make sure it handles everthing
        self.run_all(stats.midrange, lambda x: x not in [noadd])

    def test_standard_deviation(self):
        # test to make sure it handles everthing
        self.run_all(stats.standard_deviation, lambda x: x not in [noadd])

    def test_pstdev(self):
        # test to make sure it handles everthing
        self.run_all(stats.population_standard_deviation, lambda x: x not in [noadd])

    def test_variance(self):
        # test to make sure it handles everthing
        self.run_all(stats.variance, lambda x: x not in [noadd])

    def test_pvariance(self):
        # test to make sure it handles everthing
        self.run_all(stats.pvariance, lambda x: x not in [noadd])

    def test_range(self):
        # test to make sure it handles everthing
        self.run_all(stats.range, lambda x: x not in [noadd])

    def test_iqr(self):
        # test to make sure it handles everthing
        self.run_all(stats.iqr, lambda x: x not in [noadd])

    def test_average_deviation(self):
        # test to make sure it handles everthing
        self.run_all(stats.average_deviation, lambda x: x not in [noadd])

    def test_corr(self):
        # test to make sure it handles everthing
        self.run_all(stats.corr, lambda x: x not in [noadd])

    def test_cov(self):
        # test to make sure it handles everthing
        self.run_all(stats.cov, lambda x: x not in [noadd])

    def test_pcov(self):
        # test to make sure it handles everthing
        self.run_all(stats.pcov, lambda x: x not in [noadd])

    def test_errsumsq(self):
        # test to make sure it handles everthing
        self.run_all(stats.errsumsq, lambda x: x not in [noadd])

    def test_linr(self):
        # test to make sure it handles everthing
        self.run_all(stats.linr, lambda x: x not in [noadd])

    def test_sum(self):
        # test to make sure it handles everthing
        self.run_all(stats.sum, lambda x: x not in [noadd])

    def test_sumsq(self):
        # test to make sure it handles everthing
        self.run_all(stats.sumsq, lambda x: x not in [noadd])

    def test_product(self):
        # test to make sure it handles everthing
        self.run_all(stats.product, lambda x: x not in [noadd])

    def test_xsums(self):
        # test to make sure it handles everthing
        self.run_all(stats.xsums, lambda x: x not in [noadd])

    def test_xysums(self):
        # test to make sure it handles everthing
        self.run_all(stats.xysums, lambda x: x not in [noadd])

    def test_Sxx(self):
        # test to make sure it handles everthing
        self.run_all(stats.Sxx, lambda x: x not in [noadd])

    def test_Syy(self):
        # test to make sure it handles everthing
        self.run_all(stats.Syy, lambda x: x not in [noadd])

    def test_Sxy(self):
        # test to make sure it handles everthing
        self.run_all(stats.Sxy, lambda x: x not in [noadd])

    def test_sterrmean(self):
        # test to make sure it handles everthing
        self.run_all(stats.stderrmean, lambda x: x not in [noadd])


class PrecisionTests(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
