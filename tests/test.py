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

from stats import stats

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

# for better debugging output
names = [   'sequence', 'vanishing_sequence', 'pi_multiples', 'e_multiples',
            'inv_pi_multiples', 'inv_e_multiples', 'zeros', 'ones', 'noadd',
            'nomul', 'nodiv', 'precision_sequence', 'semiprecision_sequence']

#------------------------------------------------------------------------------#
#                                  Tests                                       #
#------------------------------------------------------------------------------#

class UsabilityTests(unittest.TestCase):

    def run_with_datasets(self, f, datasets, my_names):
        for pos, dataset in enumerate(datasets):
            try:
                f(dataset)
            except:
                fname = f.__name__
                setname = my_names[pos % (len(datasets)//3)]
                self.fail("%s failed on %s" % (fname, setname))

    def run_all(self, f, f_filter):
        my_names = []
        list_seqs = []
        i = 0
        for seq in seqs:
            if f_filter(seq):
                list_seqs.append(seq)
                my_names.append(names[i])
            i += 1
        tuple_seqs = [tuple(i) for i in list_seqs]
        set_seqs = [set(i) for i in list_seqs]
        my_seqs = list_seqs + tuple_seqs + set_seqs
        self.run_with_datasets(f, my_seqs, my_names)

    def run_all_bivariate(self, f, f_filter):
        my_names = []
        pre_seqs = []
        i = 0
        for seq in seqs:
            if f_filter(seq):
                pre_seqs.append(seq)
                my_names.append(names[i])
            i += 1
        list_seqs = [list(zip(x, reversed(x))) for x in pre_seqs]
        tuple_seqs = [tuple(i) for i in list_seqs]
        set_seqs = [set(i) for i in list_seqs]
        my_seqs = list_seqs + tuple_seqs + set_seqs
        self.run_with_datasets(f, my_seqs, my_names)        

    def test_mean(self):
        self.run_all(stats.mean, lambda x: x != noadd)

    def test_harmonic_mean(self):
        self.run_all(stats.harmonic_mean, lambda x: x != noadd)

    def test_geometric_mean(self):
        # XXX we get an overflow on sequence, pi_multiples, etc
        self.run_all(stats.geometric_mean, lambda x: x not in [noadd, nomul, sequence, pi_multiples, e_multiples, precision_sequence])

    def test_quadratic_mean(self):
        self.run_all(stats.quadratic_mean, lambda x: x != nomul)

    def test_median(self):
        self.run_all(stats.median, lambda x: x not in [noadd])

    def test_mode(self):
        self.run_all(stats.mode, lambda x: x not in [sequence, vanishing_sequence, pi_multiples, e_multiples, inv_pi_multiples, inv_e_multiples, precision_sequence])

    def test_midrange(self):
        self.run_all(stats.midrange, lambda x: x not in [noadd])

    def test_standard_deviation(self):
        self.run_all(stats.stdev, lambda x: x not in [noadd, ones, semiprecision_sequence])

    def test_pstdev(self):
        self.run_all(stats.pstdev, lambda x: x not in [noadd, ones, semiprecision_sequence])

    def test_variance(self):
        # variance requires addition and that there be a well-defined variance.
        # there's also a domain error that we get with the semiprecision, which
        # is at first glance suspicious
        self.run_all(stats.variance, lambda x: x not in [noadd, ones, semiprecision_sequence])

    def test_pvariance(self):
        self.run_all(stats.pvariance, lambda x: x not in [noadd])

    def test_range(self):
        self.run_all(stats.range, lambda x: x not in [noadd])

    def test_iqr(self):
        self.run_all(stats.iqr, lambda x: x not in [noadd])

    def test_average_deviation(self):
        self.run_all(stats.average_deviation, lambda x: x not in [noadd])

    def test_corr(self):
        self.run_all_bivariate(stats.corr, lambda x: x not in [noadd, e_multiples, ones, semiprecision_sequence])

    def test_cov(self):
        self.run_all_bivariate(stats.cov, lambda x: x not in [noadd, ones, semiprecision_sequence])

    def test_pcov(self):
        self.run_all_bivariate(stats.pcov, lambda x: x not in [noadd])

    def test_errsumsq(self):
        self.run_all_bivariate(stats.errsumsq, lambda x: x not in [noadd, ones, semiprecision_sequence])

    def test_linr(self):
        self.run_all_bivariate(stats.linr, lambda x: x not in [noadd, ones, semiprecision_sequence, zeros])

    def test_sum(self):
        self.run_all(stats.sum, lambda x: x not in [noadd])

    def test_sumsq(self):
        self.run_all(stats.sumsq, lambda x: x not in [noadd])

    def test_product(self):
        self.run_all(stats.product, lambda x: x not in [noadd])

    def test_xsums(self):
        self.run_all(stats.xsums, lambda x: x not in [noadd])

    def test_xysums(self):
        self.run_all_bivariate(stats.xysums, lambda x: x not in [noadd])

    def test_Sxx(self):
        self.run_all(stats.Sxx, lambda x: x not in [noadd])

    def test_Syy(self):
        self.run_all(stats.Syy, lambda x: x not in [noadd])

    def test_Sxy(self):
        self.run_all_bivariate(stats.Sxy, lambda x: x not in [noadd])

    def test_sterrmean(self):
        #self.run_all(stats.sterrmean, lambda x: x not in [noadd, sequence, vanishing_sequence, pi_multiples, e_multiples])
        pass

class PrecisionTests(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
