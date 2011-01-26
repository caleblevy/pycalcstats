#!/usr/bin/env python3

##  Copyright (c) 2011 Steven D'Aprano.
##  See the file stats/__init__.py for the licence terms for this software.

"""
Run the test suite for the stats package.

"""

#import doctests
import gc
import os
import sys
import time
import unittest


def setup():
    gc.collect()
    assert not gc.garbage


def run_doctests():  # FIXME
    print("WARNING: skipping doctests.")
    return
    failures, tests = doctest.testmod(stats)
    if failures:
        print("Skipping further tests while doctests failing.")
        sys.exit(1)
    else:
        print("Doctests: failed %d, attempted %d" % (failures, tests))


def run_example_tests():
    if os.path.exists('examples.txt'):
        failures, tests = doctest.testfile('examples.txt')
        if failures:
            print("Skipping further tests while example doctests failing.")
            sys.exit(1)
        else:
            print("Example doc tests: failed %d, attempted %d" % (failures, tests))
    else:
        print('WARNING: No example text file found.')


def run_unittests():
    t0 = time.time()
    total = failures = errors = skipped = 0
    # Tests to run:
    import stats._tests.basic
    import stats._tests.co
    import stats._tests.general
    import stats._tests.multivar
    import stats._tests.order
    import stats._tests.univar
    import stats._tests.utils
    modules = (
        stats._tests.basic,
        stats._tests.co,
        stats._tests.general,
        stats._tests.multivar,
        stats._tests.order,
        stats._tests.univar,
        stats._tests.utils,
        )
    for module in modules:
        print("\n+++ Testing module %s +++" % module.__name__)
        x = unittest.main(exit=False, module=module).result
        total += x.testsRun
        failures += len(x.failures)
        errors += len(x.errors)
        skipped += len(x.skipped)
        #suite = unittest.TestLoader().loadTestsFromTestCase(TestSequenceFunctions)
        #unittest.TextTestRunner(verbosity=2).run(suite)
        #unittest.TextTestRunner(verbosity=0).run(doc_test_suite)
    t = time.time() - t0
    print("\n" + "*"*70 + "\n")
    print("+++ Summary +++\n")
    print("Ran %d tests in %d modules in %.3f seconds:"
          % (total, len(modules), t))
    print("%d failures, %d errors, %d skipped.\n"
          % (failures, errors, skipped))
    return modules


def run_garbage_detector(make_garbage=False):
    # Simple garbage detector.
    if make_garbage:
        # Force a cycle that will be detected.
        class Junk:
            def __del__(self):
                pass
        t = Junk()
        t.rubbish = t
        del t
    gc.collect()
    if gc.garbage:
        print("List of uncollectable garbage:")
        print(gc.garbage)
    else:
        print("No garbage found.")


if __name__ == '__main__' and __package__ is not None:
    setup()
    run_doctests()
    run_example_tests()
    run_unittests()
    run_garbage_detector()

