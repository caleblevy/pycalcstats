#!/usr/bin/env python3

##  Copyright (c) 2011 Steven D'Aprano.
##  See the file stats/__init__.py for the licence terms for this software.

"""
Run the test suite for the stats package.

"""


import unittest




if __name__ == '__main__' and __package__ is not None:
    import stats.tests.utils, stats.tests.basic
    modules = (stats.tests.utils, stats.tests.basic,
    )
    total = failures = errors = skipped = 0
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
    print("\n" + "*"*70 + "\n")
    print("+++ Summary +++\n")
    print("Ran %d tests in %d modules: %d failures, %d errors, %d skipped.\n"
          % (total, len(modules), failures, errors, skipped))

