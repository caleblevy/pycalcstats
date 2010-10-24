#! /usr/bin/env python3

from distutils.core import setup

setup(  name="stats.py",
	    version="0.1a",
    	description="Elementary statistics module for Python",
	    author="Steven D'Aprano",
	    author_email="steve+python@pearwood.info",
	    url="http://pypi.python.org/pypi/stats/0.1a",
	    package_dir={'': 'src'},
        packages=['stats']
)
