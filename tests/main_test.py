#!/usr/bin/env python3
"""Test suite for stats.py

Runs:
    doctests from the stats module
    tests from the examples text file (if any)
    unit tests in this module
    a limited test for uncollectable objects

"""

import doctest
import gc
import itertools
import os
import random
import sys
import unittest


# Module being tested.
import stats

# Reminder to myself that this has to be run under Python3.
if sys.version < "3.0":
    raise RuntimeError("run this under Python3")


# === Helper functions ===

# === Data sets for testing ===

# === Test suites ===

class GlobalTest(unittest.TestCase):
    """Test the state and/or existence of globals."""
    def testSum(self):
        self.assert_(stats._sum is sum)
    def testMeta(self):
        """Test existence of metadata."""
        attrs = ("__doc__ __version__ __date__ __author__"
                 " __author_email__ __all__").split()
        for meta in attrs:
            self.failUnless(hasattr(stats, meta), "missing %s" % meta)


class MinmaxTest(unittest.TestCase):
    """Tests for minmax function."""
    data = list(range(100))
    expected = (0, 99)

    def key(self, n):
        # Tests assume this is a monotomically increasing function.
        return n*33 - 11

    def setUp(self):
        random.shuffle(self.data)

    def testArgsNoKey(self):
        """Test minmax works with multiple arguments and no key."""
        self.assertEquals(stats.minmax(*self.data), self.expected)

    def testSequenceNoKey(self):
        """Test minmax works with a single sequence argument and no key."""
        self.assertEquals(stats.minmax(self.data), self.expected)

    def testIterNoKey(self):
        """Test minmax works with a single iterator argument and no key."""
        self.assertEquals(stats.minmax(iter(self.data)), self.expected)

    def testArgsKey(self):
        """Test minmax works with multiple arguments and a key function."""
        result = stats.minmax(*self.data, key=self.key)
        self.assertEquals(result, self.expected)

    def testSequenceKey(self):
        """Test minmax works with a single sequence argument and a key."""
        result = stats.minmax(self.data, key=self.key)
        self.assertEquals(result, self.expected)

    def testIterKey(self):
        """Test minmax works with a single iterator argument and a key."""
        it = iter(self.data)
        self.assertEquals(stats.minmax(it, key=self.key), self.expected)

    def testCompareNoKey(self):
        """Test minmax directly against min and max built-ins."""
        data = random.sample(range(-5000, 5000), 300)
        expected = (min(data), max(data))
        result = stats.minmax(data)
        self.assertEquals(result, expected)
        random.shuffle(data)
        result = stats.minmax(iter(data))
        self.assertEquals(result, expected)

    def testCompareKey(self):
        """Test minmax directly against min and max built-ins with a key."""
        letters = list('abcdefghij')
        random.shuffle(letters)
        assert len(letters) == 10
        data = [count*letter for (count, letter) in enumerate(letters)]
        random.shuffle(data)
        expected = (min(data, key=len), max(data, key=len))
        result = stats.minmax(data, key=len)
        self.assertEquals(result, expected)
        random.shuffle(data)
        result = stats.minmax(iter(data), key=len)
        self.assertEquals(result, expected)


class SortedDataDecoratorTest(unittest.TestCase):
    """Test that the sorted_data decorator works correctly."""
    def testDecorator(self):
        @stats.sorted_data
        def f(data):
            return data

        values = random.sample(range(1000), 100)
        result = f(values)
        self.assertEquals(result, sorted(values))




# ============================================================================

if __name__ == '__main__':
    # Define a function that prints, or doesn't, according to whether or not
    # we're in (slightly) quiet mode. Note that we always print "skip" and
    # failure messages.
    # FIX ME can we make unittest run silently if there are no errors?
    if '-q' in sys.argv[1:]:
        def pr(s):
            pass
    else:
        def pr(s):
            print(s)
    #
    # Now run the tests.
    #
    gc.collect()
    assert not gc.garbage
    #
    # Run doctests in the stats module.
    #
    failures, tests = doctest.testmod(stats)
    if failures:
        print("Skipping further tests while doctests failing.")
        sys.exit(1)
    else:
        pr("Module doc tests: failed %d, attempted %d" % (failures, tests))
    #
    # Run doctests in the example text file.
    #
    if os.path.exists('examples.txt'):
        failures, tests = doctest.testfile('examples.txt')
        if failures:
            print("Skipping further tests while doctests failing.")
            sys.exit(1)
        else:
            pr("Example doc tests: failed %d, attempted %d" % (failures, tests))
    else:
        pr('WARNING: No example text file found.')
    #
    # Run unit tests.
    #
    pr("Running unit tests:")
    try:
        unittest.main()
    except SystemExit:
        pass
    #
    # Check for reference leaks.
    #
    gc.collect()
    if gc.garbage:
        print("List of uncollectable garbage:")
        print(gc.garbage)
    else:
        pr("No garbage found.")


