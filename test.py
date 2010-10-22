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

import stats

#------------------------------------------------------------------------------#
#                    Arithmetically Inconvenient Types                         #
#------------------------------------------------------------------------------#

class NoAddition(int):
    
    def __add__(self, other):
        raise NotImplementedError("No addition!")

    def __radd__(self, other):
        raise NotImplementedError("No addition!")


class NoMultiplication(int):

    def __mul__(self, other):
        raise NotImplementedError("No multiplication!")

    def __rmul__(self, other):
        raise NotImplementedError("No multiplication!")


class NoDivision(int):

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
semiprecision_sequence = [0.123456789012345] * 10000000
list_seqs = [   sequence,
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
# tuple versions of the above
tuple_seqs = [tuple(i) for i in list_seqs]
# set versions of the above
set_seqs = [set(i) for i in list_seqs]
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
all_seqs = list_seqs + tuple_seqs + set_seqs

#------------------------------------------------------------------------------#
#                                  Tests                                       #
#------------------------------------------------------------------------------#

class UsabilityTests(unittest.TestCase):

    def run_with_datasets(self, f, datasets):
        for dataset in datasets:
            f(dataset)

    def test_mean(self):
        # test to make sure it handles everthing except the noadd
        seqs = filter(lambda x: x != noadd, list_seqs)
        self.run_with_datasets(stats.mean, seqs)

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
