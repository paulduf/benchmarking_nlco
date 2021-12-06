#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 16:08:44 2020

@author: pauldufosse

Optimization standard routines that may be called before, after,
or in couple with a solver, e.g.
 - Differences approximation
 - Projection operator (feasibility solver)
 - Initialization procedures

TODO:
    - better handle of bounds for only lower or upper
"""
import numpy as np
from scipy.optimize import fmin_slsqp


def ffd(func, x, fx=None, eps=1e-10):
    """
    Forward-mode Finite Differences
    """
    if fx is None:
        fx = func(x)
    X = np.eye(len(x)) * eps + x
    fex = np.asarray(list(map(func, X)))
    return (fex - fx) / eps


def repair(p, problem):
    """
    Projection operator
    Finds closest feasible point x from p, by minimizing:
    norm(x - p)^2_2 st. original (nonlinear + bound) constraints
    using SciPy SLSQP
    """

    def distance(x):
        return sum((x - p)**2)

    def constraint(x):
        return np.concatenate(
            (- problem.g(x), - problem.g_bounds(x))
        )

    res = fmin_slsqp(
        distance, p,
        f_ieqcons=constraint,
        full_output=True,
        iprint=0
    )
    return res


def initialization_arnold2016(problem):
    U = np.random.uniform(size=problem.dim)
    p = problem.bounds[1, :] * U - problem.bounds[0, :]
    if any(np.isinf(p)) or any(np.isnan(p)):
        raise ValueError("Some bounds are infinite and/or undefined")
    return repair(p, problem).x
