#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 16:14:44 2021

@author: pauldufosse

Timing of the different suites of test problems

TODO
"""
from codetiming import Timer
import numpy as np
from .problems import Arnold2017

if __name__ == "__main__":

    print("Test for Arnold 2017 problem")
    for dim in [5, 10, 20, 40, 100, 200, 400, 1000, 2000, 4000, 10000]:
        m = int(dim / 2)
        print(f"Running dimension {dim} and {m} constraints", end="...")
        problem = Arnold2017(dim=dim, m=m)
        x = np.random.randn(dim)
        with Timer(f"Arnold 2017, dim {dim}"):
            for i in range(10000):
                problem.f(x)
                problem.g(x)
