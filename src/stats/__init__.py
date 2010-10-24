#!/usr/bin/env python3

##  Package stats.py
##
##  Copyright (c) 2010 Steven D'Aprano.
##
##  Permission is hereby granted, free of charge, to any person obtaining
##  a copy of this software and associated documentation files (the
##  "Software"), to deal in the Software without restriction, including
##  without limitation the rights to use, copy, modify, merge, publish,
##  distribute, sublicense, and/or sell copies of the Software, and to
##  permit persons to whom the Software is furnished to do so, subject to
##  the following conditions:
##
##  The above copyright notice and this permission notice shall be
##  included in all copies or substantial portions of the Software.
##
##  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
##  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
##  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
##  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
##  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
##  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
##  SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


"""
'Scientific calculator' statistics for Python 3.

>>> mean([-1.0, 2.5, 3.25, 5.75])
2.625
>>> data = iter([-1.0, 2.5, 3.25, 5.75, 0.0, 3.75])
>>> mean(data)
2.375


TO DO:
Unless otherwise noted, all statistical functions will operate on data in
a single pass. This allows them to process data sets that are too large to
fit in memory at once. Exceptions include functions such as median, which
will sort the data (in place, if possible) before processing.

"""

# Module metadata.
__version__ = "0.1.1a"
__date__ = "2010-10-24"
__author__ = "Steven D'Aprano"
__author_email__ = "steve+python@pearwood.info"


