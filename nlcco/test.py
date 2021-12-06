import numpy as np
from .diff import ffd
from .problems import G1


def gk(x):
    return problem.g(x)[k]

####################
# Tests G1
####################
problem = G1

# Test finite differences approximation vs gradient computation at the optimum
x = problem.xopt
assert np.allclose(G1.Df(x), ffd(G1.f, x))
for k in range(9):
    assert np.allclose(ffd(gk, x), problem.Dg(x)[k])
print("G1 test OK")

# Run SLSQP on the Arnold Testbed
# TODO
