#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 16:08:44 2020

@author: pauldufosse

TODO:
    - better handle of bounds for only lower or upper
"""
import numpy as np
from functools import wraps


class SwitchDecorator:
    """
    Parse the first argument of the decorator
    So it can be used in or outside a class
    """

    def __init__(self, in_class):
        assert isinstance(in_class, bool)
        self.in_class = int(in_class)


class Arrayize(SwitchDecorator):
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            args = list(args)
            x = args[self.in_class]
            args[self.in_class] = np.asarray(x)
            return func(*args, **kwargs)
        return wrapper


def realfunction(func):
    """
    Sanitize output of multivariate function as np.float64
    """
    @wraps(func)  # to propagate __name__
    def wrapper(*args, **kwargs):
        f = func(*args, **kwargs)
        return np.float64(f)
    return wrapper


class ConstrainedTestProblem:
    """
    Base class for Constrained Test Problem

    Keyword arguments:
        - seed: fix a random seed, e.g. to sample starting point
        - eps_feas: add (nonnegative) epsilon to inequality constraints to
        enforce strictly feasible solutions

    To be inherited
    """

    def __init__(self, seed=1, eps_feas=0):
        self.seed = seed
        self.eps_feas = eps_feas
        assert self.eps_feas >= 0

    def f(self, x):
        raise NotImplementedError

    def g(self, x):
        raise NotImplementedError

    def Df(self, x):
        raise NotImplementedError

    def Dg(self, x):
        raise NotImplementedError

    @property
    def is_ineq(self):
        raise NotImplementedError

    @property
    def dim(self):
        raise NotImplementedError

    @property
    def m(self):
        return len(self.is_ineq)

    @property
    def bounds(self):
        return None

    def __call__(self, x, add_bounds=False,):
        f_ = self.f(x)
        g_ = self.g(x)
        g_ += self.eps_feas * np.asarray(self.is_ineq)  # converts to np.array
        if add_bounds:
            g_b_ = self.g_bounds(x) + self.eps_feas
            g_ = np.concatenate((g_, g_b_))
        return f_, g_

    def sample_x_start(self):
        """
        Sample x_start from uniform distribution in the bounds
        """
        if self.dim is None:
            raise
        np.random.seed(self.seed)
        self.x_start = np.zeros(self.dim)
        if self.bounds[0] is not None:
            self.x_start = self.bounds[0] + np.random.uniform(size=self.dim) * (self.bounds[1] - self.bounds[0])
        # check if upper bound was infty and correct
        for i in range(self.dim):
            xi = self.x_start[i]
            if xi == np.inf:
                xi = self.bounds[0][i] + 10
        return self.x_start

    def g_bounds(self, x):
        """
        returns bound constraints violation in the form of twice as much inequality constraints
        """
        #todo: current version is simple but we may think about desirable behaviour when one bound is np.inf or None
        if self.bounds is None:
            raise
        return np.concatenate((self.bounds[0] - x, x - self.bounds[1]))


class RestrictedCP:
    """
    Restricted Constrained Problem
    Encapsulates a constrained problem, overwrites the constraint function,
    returns only positive values
    """

    def __init__(self, problem):
        self.problem = problem
        self.old_g = self.problem.g
        self.problem.g = self.g

    def __getattr__(self, attr):
        return getattr(self.problem, attr)

    def __call__(self, x, **kwargs):
        return self.problem.__call__(x, **kwargs)

    def g(self, x):
        g_ = np.asarray(self.old_g(x))
        return g_ * (g_ > 0)


class DemoProblem(ConstrainedTestProblem):
    """
    >>> demo = DemoProblem()
    >>> xtest = np.zeros(2)
    >>> demo(xtest)
    (0.0, array([1.]))
    >>> demo.eps_feas = 1e-3
    >>> demo(xtest)
    (0.0, array([1.001]))
    >>> demo(xtest, add_bounds=True)
    (0.0, array([ 1.001, -4.999, -4.999, -4.999, -4.999]))
    """
    dim = 2
    m = 1
    is_ineq = [True]
    bounds = np.asarray([[-5, 5]] * 2).T
    f = staticmethod(lambda x: sum(np.asarray(x)**2))
    g = staticmethod(lambda x: [1 - x[0] - x[1]])


def epstolrel(fsucc, fmin, f0):
    return abs((fsucc - fmin) / (f0 - fmin))


def feasibility_ratio(x_seq, g):
    """
    Feasibility ratio of incumbent solution over time

    x_seq: sequence of incumbent
    g: multivariate, multivalued constraint function
    """
    g_seq = np.asarray([g(x) for x in x_seq])
    feas_ratio = np.sum(g_seq > 0, axis=1)
    return feas_ratio


class BaseRunner:
    """
    Base class for running an algorithm on a given problem.
    Counts the number of f- and g-calls
    Store the best feasible individual seen so far
    Implements a self.run and a self.fitness method if needed
    """
    @staticmethod
    def print():
        print(47)

    def __init__(self, problem):
        self.problem = problem

        # Store best feasible
        self.feasible_seen = False
        self.best_feas_f = np.inf
        self.best_feas_x = np.zeros(problem.dim)
        self.best_last_found = 0

        # Monitor functions calls
        self.countf = 0
        self.countg = 0

    def _track_best(self, x, f_, g_):
        """
        to be called by the fitness method
        Stores the best feasible indiidual seen so far
        Args:
            x: incumbent
            f_: f(x)
            g_: g(x) (with bounds)

        Returns:
            Nothing
        """
        if f_ < self.best_feas_f and all(g_ <= 0):
            self.best_feas_x = x
            self.best_feas_f = f_
            self.best_last_found = 0
            self.feasible_seen = True
        elif self.feasible_seen:
            self.best_last_found += 1

    def objective(self, x):
        self.countf += 1
        return self.problem.f(x)

    def constraint(self, x):
        self.countg += 1
        return self.problem.g(x)

    def fitness(self, x):
        raise NotImplementedError

    def run(self, x):
        raise NotImplementedError