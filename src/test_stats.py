#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test code for the top-level module of the stats package."""

# FIXME Tests with random data currently cannot be replicated if they fail.
# I'm not happy with this -- it means that a test may pass nearly always,
# but occasionally fail. Because the data was random, it's near impossible
# to replicate the failure.


import collections
import functools
import itertools
import math
import operator
import random
import unittest

from decimal import Decimal
from fractions import Fraction

# The module(s) to be tested:
import stats



# === Unit tests ===

# NOTE: do not use self.fail... unit tests, as they are deprecated in
# Python 3.2. Although plural test cases such as self.testEquals and
# friends are not officially deprecated, they are discouraged.


class MetadataTest(unittest.TestCase):
    """Test for the existence of module metadata."""
    standard = ("__doc__", "__all__")
    extra = ("__version__", "__date__", "__author__", "__author_email__")
    module = stats

    def check_meta_exists(self, names):
        # Checks for the existence of metadata.
        for name in names:
            self.assertTrue(hasattr(self.module, name),
                "%s not present" % name)

    def testMeta(self):
        self.check_meta_exists(self.standard)

    def testExtraMeta(self):
        self.check_meta_exists(self.extra)

    def testDoc(self):
        # Test the module doc string exists and is an actual string.
        self.assertTrue(isinstance(self.__doc__, str))


class GlobalsTest(unittest.TestCase):
    module = stats

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


class StatsErrorTest(unittest.TestCase):
    def testHasException(self):
        self.assertTrue(hasattr(stats, 'StatsError'))
        self.assertTrue(issubclass(stats.StatsError, ValueError))



# === Run tests ===

class DocTests(unittest.TestCase):
    def testDocTests(self):
        import doctest
        failed, tried = doctest.testmod(stats)
        self.assertTrue(tried > 0)
        self.assertTrue(failed == 0)


def test_main():
    unittest.main()


if __name__ == '__main__':
    test_main()

