import numpy as np
from nlcco.optutils import initialization_arnold2016
from nlcco.problems import G1
from scipy.optimize import fmin_slsqp

n_exp = 100
problem = G1()


def constraint(x):
    return -problem.constraint(x)


successes = []

if __name__ == "__main__":
    for i in range(n_exp):
        x0 = initialization_arnold2016(problem)
        res = fmin_slsqp(
            problem.f, x0,
            f_ieqcons=constraint,
            full_output=True,
            iprint=0
        )
        if np.allclose(res.x, problem.xopt):
            successes.append(res)

    print(f"{len(successes)} successes and runlength data is \
          {[res.nfev for res in successes]}")
