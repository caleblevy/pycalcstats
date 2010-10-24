#! /usr/bin/env python3

"""
distributions.py

Written by Geremy Condra
Licensed under the Python License
Released 23 October 2010

This module contains the definitions for a variety of known distributions as
well as an abstract distribution class which users should subclass and populate
and an unknown distribution for containing arbitrary datapoints.
"""

import math

import stats


#-------------------------------- Exceptions ----------------------------------#
class DistributionError(Exception):
    """General error for when distributions go awry"""
    pass


class UpdateError(DistributionError):
    """An error meaning that we couldn't add data to a distribution"""
    pass

#------------------------------------------------------------------------------#

#------------------------------ Utility Functions -----------------------------#

def is_public_method(args):
    """Simple function that pulls public callables out of a class"""
    name, f = args
    if name.startswith('_'):
        return False
    return hasattr(f, '__call__')



def precompute(cls, update_prefix='update_', recompute_prefix='recompute_'):
    """This decorator makes it easy for us to precompute statistics.

    To be more specific, it does four things:
    
    1. It identifies all the bulk and single recompute methods for statistics.
    2. It builds a list of the members that should be managed this way.
    3. It adds those members to the class.
    4. It sets the prefixes that should be used to dispatch to these methods.

    The goal is to make it so that when a developer wants to define a new
    statistic on a distribution, they just add either a update_<varname> method,
    a recompute_<varname> method, or both. We can handle the rest from there.
    """
    # build the eventual targets
    target_vars = set()

    # get all the methods of the class
    methods = filter(is_public_method, cls.__dict__.items())

    # get all the single-element update methods
    for name, f in methods:
        # check to see if the method name starts with either of the prefixes
        if name.startswith(update_prefix):
            varname = name[len(update_prefix):]
        elif name.startswith(recompute_prefix):
            varname = name[len(recompute_prefix):]
        else:
            continue
        # if it does, then we'll add it to the set of variables to update
        target_vars.add(varname)

    # now we add a variable by each correct name to the class
    for varname in target_vars:
        setattr(cls, varname, None)

    # and add a frozen version of target_vars to the class
    cls.target_vars = tuple(target_vars)

    # and set the class's update and recompute prefixes
    cls.update_prefix = update_prefix
    cls.recompute_prefix = recompute_prefix

    # and go home
    return cls



class MethodStorage:
    """Decorator to add a function attribute of the given name and value"""

	def __init__(self, name, value):
		self.varname = name
		self.initial_value = value

	def __call__(self, f):
		setattr(f, self.varname, self.initial_value)
		return f

#------------------------------------------------------------------------------#


#----------------------------- Unknown Distributions --------------------------#

class BaseDistribution:
    """Base Distribution class"""

    # stores all the data points in the distribution
    values = None

    # stores the names of all the up-to-date variables in the distribution
    up_to_date = None

    def __init__(self, values=None):
        """Create the distribution, adding the given values to self.values"""
        self.values = []
        self.up_to_date = set()
        self.recompute(values)

    def _get_update_method(self, varname):
        """Returns the update_<varname> method if present and None otherwise."""
        try:
            return getattr(self, self.update_prefix + varname)
        except AttributeError:
            return None

    def _get_recompute_method(self, varname):
        """Returns the recompute_<varname> method if present, None otherwise."""
        try:
            return getattr(self, self.recompute_prefix + varname)
        except AttributeError:
            return None

    def update(self, value):
        """Updates stored values for this distribution using the given value.

        This method is useful where it's possible to compute some change in a
        property of a distribution more easily than it is to recompute the value
        from scratch, but will fall back on the latter strategy if it can't find
        an update method for a variable.

        This method also marks the variables it has successfully recomputed as
        clean as it goes, clearing the set initially.
        """
        # we add this to the list of values so that if we have to invoke the
        # bulk updater it will have already been updated.
        self.values.append(value)

        # mark the precomputed values as unclean
        self.up_to_date = set()

        for varname in self.target_vars:

            # get both the single and bulk updaters, just in case
            update_method = self._get_update_method(varname)
            recompute_method = self._get_recompute_method(varname)

            # try using the single value updater            
            if update_method:
                update_method(value)
            # if that doesn't work, fall back to the bulk updater
            elif recompute_method:
                recompute_method()
            # if we can't find either, fail noisily
            else:
                raise UpdateError("Could't update %s" % varname)

            # since we successfully updated the value, we'll flag it as
            # being up to date
            self.up_to_date.add(varname)

    def recompute(self, values=None):
        """Recomputes the distribution's precomputed values from scratch.

        This method is useful when you want to perform a bulk update or when
        you want to add a large number of data points at the same time without
        invoking the update() method individually.

        The optional 'values' argument, if given, should be an iterable whose
        elements are data points to add to self.values prior to recomputation.

        If this can't find an efficient bulk updater for a given variable, it
        will fall back to invoking the single value updater once for each value
        given.

        This method also marks the variables it has successfully recomputed as
        clean as it goes, clearing the set initially.
        """
        # extend the list of values with the given elements, if present
        if values: self.values.extend(values)

        # mark the precomputed values as unclean
        self.up_to_date = set()

        # iterate over the target variables, calling the bulk updater where
        # possible and falling back if necessary.
        for varname in self.target_vars:

            update_method = self._get_update_method(varname)
            recompute_method = self._get_recompute_method(varname)

            # prefer the bulk updater, unlike the above
            if recompute_method:
                recompute_method()
            # fallback position
            elif update_method:
                # note that this could be *very* slow- does this need an option?
                for value in self.values:
                    update_method(value)
            # only raised if we couldn't find either
            else:
                raise UpdateError("Couldn't recompute %s" % varname)

            # since we successfully updated the value, we'll flag it as
            # being up to date
            self.up_to_date.add(varname)


@precompute
class UnknownDistribution(BaseDistribution):
    """This class provides some basic statistics about its data points.

    The UnknownDistribution should be used when you want to model a distribution
    but have no idea what it is.

    Usage:

    >>> import stats.distribution
    >>> d = stats.distribution.UnknownDistribution([5.0, .1, 1e10])
    >>> d.sum
    10000000005.1
    >>> d.update(26)
    >>> d.sum
    10000000031.1

    """

    @MethodStorage('partials', [])
    def update_sum(self, x):
        stats.add_partial(x, self.update_sum.partials)
        self.sum = math.fsum(self.update_sum.partials)

    def recompute_sum(self):
        self.sum = math.fsum(self.values)

    @MethodStorage('partials', [])
    def update_sum_of_squares(self, x):
        x2 = x**2
        stats.add_partial(x2, self.update_sum_of_squares.partials)
        self.sum_of_squares = math.fsum(self.update_sum_of_squares.partials)

    def recompute_sum_of_squares(self):
        self.sum_of_squares = sum(x**2 for x in self.values)

    def update_square_of_sums(self, x):
        if not 'sum' in self.up_to_date:
            self.update_sum(x)
        self.square_of_sums = self.sum**2

    def recompute_square_of_sums(self):
        if not 'sum' in self.up_to_date:
            self.recompute_sum()
        self.square_of_sums = self.sum**2

    def update_variance(self, x):
        pass

    def recompute_variance(self):
        pass

    def update_standard_deviation(self, x):
        pass

    def recompute_standard_deviation(self):
        pass

    def update_range(self, x):
        pass

    def recompute_range(self):
        pass

    def update_interquartile_range(self, x):
        pass

    def recompute_interquartile_range(self):
        pass

    def update_mean(self, x):
        pass

    def recompute_mean(self):
        pass

    def update_harmonic_mean(self, x):
        pass

    def recompute_harmonic_mean(self):
        pass

    def update_geometric_mean(self, x):
        pass

    def recompute_geometric_mean(self):
        pass

    def update_quadratic_mean(self, x):
        pass

    def recompute_quadratic_mean(self):
        pass

    def update_median(self, x):
        pass

    def recompute_median(self):
        pass

    def update_mode(self, x):
        pass

    def recompute_mode(self):
        pass

    def update_midrange(self, x):
        pass

    def recompute_midrange(self):
        pass

