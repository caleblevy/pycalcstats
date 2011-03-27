#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##  Copyright (c) 2011 Steven D'Aprano.
##  See the file stats/__init__.py for the licence terms for this software.

"""Test suite for the rest of the stats package."""

import unittest

import test_stats

# Modules to test:
import stats.co
import stats.order
import stats.multivar
import stats.univar



# === Unit tests ===


class CoGlobalsTest(test_stats.GlobalsTest):
    module = stats.co

class OrderGlobalsTest(test_stats.GlobalsTest):
    module = stats.order

class MultivarGlobalsTest(test_stats.GlobalsTest):
    module = stats.multivar

class UnivarGlobalsTest(test_stats.GlobalsTest):
    module = stats.univar



# === Run tests ===

def test_main():
    # support.run_unittest(...list tests...)
    unittest.main()


if __name__ == '__main__':
    test_main()

