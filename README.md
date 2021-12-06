# benchmarking_nlco
Benchmarking Nonlinear Constrained Optimization algorithms

**This is work in progress**

## Description

There are two types of problems: fixed problems and scalable ones. A concrete problem inherites from the base classe `base.ConstrainedTestProblem`.

A fixed problem has fixed dimension `dim` and number of inequality constraints `m`. Most of its methods are static hence it can be used without instantiation. However the property `m` cannot be accessed in this class. Later improvement is to define _class properties_.

A scalable problem needs instantiation with dimension and number of constraints.

## TODOs

- Add the decorators `arrayize` and `realfunction` everywhere.
- Implement the gradient everywhere it is possible

## Install

Just do `pip install .` in this repository, everything relies on `numpy` and `scipy.optimize` if you want to run experiments.