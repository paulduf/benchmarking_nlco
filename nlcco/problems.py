#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 16:08:44 2020

@author: pauldufosse

Guidelines:
    - if f, g can be made static then they should be 
    (one may not want to instantiate a problem but only access f, g)

Particularities for each problem that are relevant for coding
    - G2, G3 are dimension free (hence an __init__ method)
    - G11 has many global optima (see data dict)

Warning:
    - dimension 1 not tested
"""
# NB: f and g are static method
from .base import ConstrainedTestProblem
import numpy as np

# Nonlinear test problems from litterature

class G1(ConstrainedTestProblem):
    """
    Test for static method and exact output
    >>> G1.f(G1.xopt)
    -15
    """
    
    is_ineq = [True] * 9
    bounds = np.asarray([[0, 1]] * 9 + [[0, 100]] * 3 + [[0, 1]]).T
    dim = 13
    xopt = [1] * 9 + [3] * 3 + [1]
    fmin = -15

    @staticmethod
    def f(x):
        x = np.asarray(x)
        return 5 * sum(x[0:3]) - 5 * sum(x[0:3]**2) - sum(x[4:])

    @staticmethod
    def g(x):
        g_ = []
        g_.append(2 * x[0] + 2 * x[1] + x[9] + x[10] - 10)
        g_.append(2 * x[0] + 2 * x[2] + x[9] + x[11] - 10)
        g_.append(2 * x[1] + 2 * x[2] + x[10] + x[11] - 10)
        g_.append(-8 * x[0] + x[9])
        g_.append(-8 * x[1] + x[10])
        g_.append(-8 * x[2] + x[11])
        g_.append(-2 * x[3] - x[4] + x[9])
        g_.append(-2 * x[5] - x[6] + x[10])
        g_.append(-2 * x[7] - x[8] + x[11])
        return np.asarray(g_)


class G2(ConstrainedTestProblem):
    """
    TODO : Dimension free implementation
    >>> G2.f([1]  * 4)
    0.10320385173263867
    """

    is_ineq = [True]
    dim = 20
    bounds = np.asarray([[0, 10]] * 20).T
    xopt = (3.16246061572185, 3.12833142812967, 3.09479212988791,
            3.06145059523469, 3.02792915885555, 2.99382606701730,
            2.95866871765285, 2.92184227312450, 0.49482511456933,
            0.48835711005490, 0.48231642711865, 0.47664475092742,
            0.47129550835493, 0.46623099264167, 0.46142004984199,
            0.45683664767217, 0.45245876903267, 0.44826762241853,
            0.44424700958760, 0.44038285956317)
    fmin = -0.80361910412559
    
        
    @staticmethod
    def f(x):
        x = np.asarray(x)
        v = np.cos(x)
        top = sum(v**4) - 2 * np.prod(v**2)
        bottom = np.sqrt(sum(np.multiply(np.arange(1, len(x) + 1), x**2)))
        return - np.abs(top / bottom)
    
    @staticmethod
    def g(x):
        g_ = []
        g_.append(0.75 - np.prod(x))
        g_.append(sum(x) - 7.5 * len(x)) # do not use self.dim to keep this static ?
        return np.asarray(g_)


class G3(ConstrainedTestProblem):
    """
    >>> G3.f([1] * 4)
    16.0
    >>> G3.g([1] * 4)
    3
    >>> G3.get_xopt(4)
    array([0.5, 0.5, 0.5, 0.5])
    """
    
    is_ineq = [False]
    dim = 10
    bounds = np.asarray([[0, 1]] * 10).T
    xopt = (0.31624357647283069, 0.316243577414338339, 0.316243578012345927,
            0.316243575664017895, 0.316243578205526066, 0.31624357738855069,
            0.316243575472949512, 0.316243577164883938, 0.316243578155920302,
            0.316243576147374916)
    fmin = -1.00050010001000

    @staticmethod
    def f(x):
        dim = len(x)
        return - np.sqrt(dim)**dim * np.prod(x)

    @staticmethod
    def g(x):
        x = np.asarray(x)
        return sum(x**2) - 1

    @staticmethod
    def get_xopt(dim):
        return np.asarray([1 / np.sqrt(dim)] * dim)

    
class G4(ConstrainedTestProblem):
    """
    >>> G4.f(G4.x_start)
    -25147.493180000005
    >>> G4.g(G4.x_start) > 0
    array([False,  True, False, False, False, False])
    >>> G4.f((78,33,29.995,45,36.776))
    -30665.608767818834
    """

    is_ineq = [True] * 6
    bounds = np.asarray([[78, 102], [33, 45]] + [[27, 45]]*3).T
    dim = 5
    xopt = (78, 33, 29.9952560256815985, 45, 36.7758129057882073)
    fmin = -30665.53867178332
    x_start = [100] + [40]*4

    @staticmethod
    def f(x):
        return 5.3578547 * x[2]**2 + 0.8356891 * x[0] * x[4] + 37.293239 * x[0] - 40792.141

    @staticmethod
    def g(x):
        h1 = 85.334407 + 0.0056858 * x[1] * x[4] + 0.0006262 * x[0] * x[3] - 0.0022053 * x[2] * x[4]
        h2 = 80.51249 + 0.0071317 * x[1] * x[4] + 0.0029955 * x[0] * x[1] + 0.0021813 * x[2]**2
        h3 = 9.300961 + 0.0047026 * x[2] * x[4] + 0.0012547 * x[0] * x[2] + 0.0019085 * x[2] * x[3]
        return np.asarray([-h1, h1 - 92, 90 - h2, h2 - 110, 20 - h3, h3 - 25])


class G5(ConstrainedTestProblem):
    """
    >>> G5.f((679.9453, 1026.067, 0.1188764, 0.3962336))
    5126.497478059328
    """
     
    is_ineq = [True] * 2 + [False] * 3
    bounds = np.asarray([[0, 1200]] * 2 + [[-0.55, 0.55]] * 2).T
    dim = 4
    xopt = (679.945148297028709, 1026.06697600004691, 0.118876369094410433,
            -0.39623348521517826)
    fmin = 5126.4967140071
    
    @staticmethod
    def f(x):
        return 3 * x[0] + 1e-6 * x[0]**3 + 2 * x[1] + 2e-6 / 3 * x[1]**3
    
    @staticmethod
    def g(x):
        g_ = []
        g_.append(x[2] - x[3] - 0.55)
        g_.append(-x[2] + x[3] - 0.55)
        g_.append(1e3 * np.sin(-x[2] - 0.25) + 1e3 * np.sin(-x[3] - 0.25) + 894.8 - x[0])
        g_.append(1e3 * np.sin(x[2] - 0.25) + 1e3 * np.sin(x[2] - x[3] - 0.25) + 894.8 - x[1])
        g_.append(1e3 * np.sin(x[3] - 0.25) + 1e3 * np.sin(x[3] - x[2] - 0.25) + 1294.8)
        return np.asarray(g_)


class G6(ConstrainedTestProblem):
    """
    >>> G6.f((14.095,0.84296))
    -6961.814744487831
    """

    is_ineq = [True] * 2
    bounds = np.asarray([[13, 100], [0, 100]]).T
    dim = 2
    x_start = [20.1, 5.84]
    xopt = (14.09500000000000064, 0.8429607892154795668)
    fmin = -6961.81387558015

    @staticmethod
    def f(x):
        return (x[0] - 10)**3 + (x[1] - 20)**3

    @staticmethod
    def g(x):
        return [- (x[0] - 5)**2 - (x[1] - 5)**2 + 100, 
                (x[0] - 6)**2 + (x[1] - 5)**2 - 82.81]


class G7(ConstrainedTestProblem):
    """
    >>> G7.f((2.171996, 2.363683,8.773926,5.095984,0.9906548,1.430574,1.321644,9.828726,8.280092,8.375927))
    24.30620316945705
    """

    is_ineq = [True] * 8
    bounds = np.asarray([[-10, 10]] * 10).T
    xopt = (2.17199634142692, 2.3636830416034, 8.77392573913157, 5.09598443745173, 
            0.990654756560493, 1.43057392853463, 1.32164415364306,
            9.82872576524495, 8.2800915887356, 8.3759266477347)
    fmin = 24.30620906818
    dim = 10

    @staticmethod
    def f(x):
        v = x[0]**2 + x[1]**2 + x[0]*x[1] - 14*x[0] - 16*x[1] 
        v += (x[2] - 10)**2
        v += 4 * (x[3] - 5)**2
        v += (x[4] - 3)**2
        v += 2 * (x[5] - 1)**2
        v += 5 * x[6]**2
        v += 7 * (x[7] - 11)**2
        v += 2 * (x[8] - 10)**2
        v += (x[9] - 7)**2
        v += 45
        return v

    @staticmethod
    def g(x):
        g_ = []
        g_.append(-105 + 4 * x[0] + 5 * x[1] - 3 * x[6] + 9 * x[7])
        g_.append(10 * x[0] - 8 * x[1] - 17 * x[6] + 2 * x[7])
        g_.append(-8 * x[0] + 2 * x[1] + 5 * x[8] - 2 * x[9] - 12)
        g_.append(-3 * x[0] + 6 * x[1] + 12 * (x[8] - 8)**2 - 7 * x[9])
        g_.append(3 * (x[0] - 2)**2 + 4 * (x[1] - 3)**2 + 2 * x[2]**2 - 7 * x[3] - 120)
        g_.append(x[0]**2 + 2 * (x[1] - 2)**2 - 2 * x[0] * x[1] + 14 * x[4] - 6 * x[5])
        g_.append(5 * x[0]**2 + 8 * x[1] + (x[2] - 6)**2 - 2 * x[3] - 40)
        g_.append(.5 * (x[0] - 8)**2 + 2 * (x[1] - 4)**2 + 3 * x[4]**2 - x[5] - 30)
        return np.asarray(g_)


class G8(ConstrainedTestProblem):

    is_ineq = [True] * 2
    bounds = np.asarray([[0, 10]] * 2).T
    dim = 2
    xopt = (1.22797135260752599, 4.24537336612274885)
    fmin = -0.0958250414180359

    @staticmethod
    def f(x):
        v = np.sin(2 * np.pi * x[0])**3 * np.sin(2 * np.pi * x[1])
        v /= x[0]**3 * (sum(x)) 
        return -v

    @staticmethod
    def g(x):
        g_ = []
        g_.append(x[0]**2 - x[1] + 1)
        g_.append(1 - x[0] + (x[1] - 4)**2)
        return np.asarray(g_)


class G9(ConstrainedTestProblem):
    """
    >>> G9.f((2.330499, 1.951372, -0.4775414, 4.365726, -0.624487, 1.038131, 1.594227))
    680.6301112407558
    """
   
    is_ineq = [True] * 4
    bounds = np.asarray([[-10, 10]]  * 7).T
    xopt = (2.33049935147405174, 1.95137236847114592, -0.477541399510615805,
            4.36572624923625874, -0.624486959100388983, 1.03813099410962173,
            1.5942266780671519)
    fmin = 680.630057374402
    dim = 7
  
    @staticmethod
    def f(x):
        v = (x[0] - 10)**2
        v += 5 * (x[1] - 12)**2
        v += x[2]**4
        v += 3 * (x[3] - 11)**2
        v += 10 * x[4]**6
        v += 7 * x[5]**2
        v += x[6]**4
        v += -4 * x[5] * x[6]
        v += -10 * x[5]
        v += -8 * x[6]
        return v

    @staticmethod
    def g(x):
        g_ = []
        g_.append(-127 + 2 * x[0]**2 + 3 * x[1]**4 + x[2] + 4 * x[3]**2 + 5 * x[4])
        g_.append(-196 + 23 * x[0] + x[1]**2 + 6 * x[5]**2 - 8 * x[6])
        g_.append(-282 + 7 * x[0] + 3 * x[1] + 10 * x[2]**2 + x[3] - x[4])
        g_.append(4 * x[0]**2 + x[1]**2 - 3 * x[0] * x[1] + 2 * x[2]**2 + 5 * x[5] - 11 * x[6])
        return np.asarray(g_)


class G10(ConstrainedTestProblem):
    """
    >>> G10.f((579.3167,1359.943,5110.071,182.0174,295.5985,217.9799,286.4162,395.5979))
    7049.3307
    """

    is_ineq = [True] * 6
    bounds = np.asarray([[1e2, 1e4]] + [[1e3, 1e4]] * 2 + [[1e1, 1e3]] * 5).T
    xopt = (579.306685017979589, 1359.97067807935605, 5109.97065743133317, 
            182.01769963061534, 295.601173702746792, 217.982300369384632,
            286.41652592786852, 395.601173702746735)
    x_start = [6881.97594858, 1352.20925325, 9699.06535149,  636.413402 ,304.48287295,  979.73387336,  846.3647499 ,  693.43198988]
    fmin = 7049.24802052867
    dim = 8

    @staticmethod
    def f(x):
        return x[0] + x[1] + x[2]

    @staticmethod
    def g(x):
        g_ = []
        g_.append(-1 + .0025 * (x[3] + x[5]))
        g_.append(-1 + .0025 * (x[4] + x[6] - x[3]))
        g_.append(-1 + .01 * (x[7] - x[4]))
        g_.append(- x[0] * x[5] + 833.33252 * x[3] + 100 * x[0] - 83333.333)
        g_.append(- x[1] * x[6] + 1250 * x[4] + x[1] * x[3] - 1250 * x[3])
        g_.append(- x[2] * x[7] + 1250000 + x[2] * x[4] - 2500 * x[4])
        return np.asarray(g_)


class G11(ConstrainedTestProblem):
    """
    >>> G11.f((0.70711, 0.5))
    0.7500045521
    """
    
    is_ineq = [False]
    bounds = np.asarray([[-1, 1]] * 2).T
    xopt = (-0.707036070037170616, 0.500000004333606807)
    fmin = 0.7499
    dim = 2
    
    @staticmethod
    def f(x):
        return x[0]**2 + (x[1] - 1)**2

    @staticmethod
    def g(x):
        return np.asarray([x[1] - x[0]**2])
    

class G12(ConstrainedTestProblem):
    """
    >>> G12.f(5 * np.ones(3))
    -1.0
    >>> G12.g(G12.xopt)
    array([-0.0625, -0.0625, -0.0625])
    """
    is_ineq = [True]
    bounds = np.asarray([[0, 10]] * 3).T
    dim = 3
    fmin = -1
    xopt = [5] * 3
    
    @staticmethod
    def f(x):
        x = np.asarray(x)
        return -(100 - sum((x - 5)**2)) / 100
    
    @staticmethod
    def g(x):
        # check if equivalent: round to closest integer
        x_ = np.round(x)
        return np.asarray((x - x_)**2 - .0625)
    
    
class G13(ConstrainedTestProblem):
    
    is_ineq = [False] * 3
    bounds = np.asarray([[-2.3, 2.3]] * 2 + [[-3.2, 3.2]] * 3).T
    dim = 5
    xopt = (-1.71714224003, 1.59572124049468, 1.8272502406271,
            -0.763659881912867, -0.76365986736498)
    fmin = 0.053941514041898

    @staticmethod
    def f(x):
        return np.exp(np.prod(x))
    
    @staticmethod
    def g(x):
        x = np.asarray(x)
        g_ = []
        g_.append(sum(x**2 - 10))
        g_.append(x[1] * x[2] - 5 * x[3] * x[4])
        g_.append(x[0]**3 + x[1]**3 + 1)
        return np.asarray(g_)
    

class G14(ConstrainedTestProblem):
    """
    >>> G14.f((0.0406684113216282, 0.147721240492452, 0.783205732104114, 0.00141433931889084, 0.485293636780388, 0.000693183051556082, 0.0274052040687766, 0.0179509660214818, 0.0373268186859717, 0.0968844604336845))
    -47.764888459491466
    """
    
    is_ineq = [False] * 3
    bounds = np.asarray([[0, 10]] * 10)
    dim = 10
    xopt = (0.0406684113216282, 0.147721240492452, 0.783205732104114,
            0.00141433931889084, 0.485293636780388, 0.000693183051556082,
            0.0274052040687766, 0.0179509660214818, 0.0373268186859717,
            0.0968844604336845)
    fmin = -47.7648884594915
    
    @staticmethod
    def f(x):
        c = np.asarray([-6.089, -17.164, -34.054, -5.914, -24.721,
         -14.986, -24.1, -10.708, -26.662, -22.179])
        f_ = np.log(x)
        f_ -= np.log(sum(x))
        f_ += c
        return sum(np.multiply(x, f_))
    
    @staticmethod
    def g(x):
        pass
    
class G15(ConstrainedTestProblem):
    
    is_ineq = [False] * 2
    bounds = np.asarray([[0, 10]] * 3)
    dim = 3
    xopt = (3.51212812611795133, 0.216987510429556135, 3.55217854929179921)
    fmin = 961.715022289961
    
    @staticmethod
    def f(x):
        x = np.asarray(x)
        #return 1000 - sum(np.multiply([1, 2, 1], x**2)) - x[0] * x[1] - x[1] * x[2]
        return 1000 - x[0]**2 - 2 * x[1]**2 - x[2]**2 - x[0] * x[1] - x[1] * x[2]
    
    @staticmethod
    def g(x):
        x = np.asarray(x)
        g_ = []
        g_.append(sum(x**2) - 25)
        g_.append(np.multiply([8, 14, 7], x) - 56)
        return np.asarray(g_)


class G16(ConstrainedTestProblem):
    """
    >>> xopt = G16.xopt
    >>> G16.f(xopt)
    -1.9051552585347862
    >>> sum(G16.g(xopt) > 0)
    0
    """
    
    dim = 5
    is_ineq = [True] * 38
    bounds = np.array([[704.4148, 906.3855], [68.6, 288.88], [0, 134.75], 
                       [193, 287.0966], [25, 84.1988]]).T
    xopt = (705.174537070090537, 68.5999999999999943, 102.899999999999991, 
            282.324931593660324, 37.5841164258054832)
    fmin = -1.90515525853479
 
    @staticmethod
    def get_vals(x):
        y = np.zeros(17)
        c = np.zeros(17)
        y[0] = x[1] + x[2] + 41.6
        c[0] = 0.024 * x[3] - 4.62
        y[1] = 12.5 / c[0] + 12
        c[1] = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y[1] * x[0]
        c[2] = 0.052 * x[0] + 78 + 0.002377 * y[1] * x[0]
        y[2] = c[1] / c[2]
        y[3] = 19 * y[2]
        c[3] = 0.04782 * (x[0] - y[2]) + 0.1956 * (x[0] - y[2])**2 / x[1] + 0.6376 * y[3] + 1.594 * y[2]
        c[4] = 100 * x[1]
        c[5] = x[0] - y[2] - y[3]
        c[6] = 0.950 - c[3] / c[4]
        y[4] = c[5] * c[6]
        y[5] = x[0] - y[4] - y[3] - y[2]
        c[7] = 0.995 * (y[4] + y[3])
        y[6] = c[7] / y[0]
        y[7] = c[7] / 3798
        c[8] = y[6] - 0.0663 * y[6] / y[7] - 0.3153
        y[8] = 96.82 / c[8] + 0.321 * y[0]
        y[9] = 1.29 * y[4] + 1.258 * y[3] + 2.29 * y[2] + 1.71 * y[5]
        y[10] = 1.71 * x[0] - 0.452 * y[3] + 0.580 * y[2]
        c[9] = 12.3 / 752.3
        c[10] = 1.74125 * y[1] * x[0]
        c[11] = 0.995 * y[9] + 1998
        y[11] = c[9] * x[0] + c[10] / c[11]
        y[12] = c[11] - 1.75 * y[1]
        y[13] = 3623 + 64.4 * x[1] + 58.4 * x[2] + 146312 / (y[8] + x[4])
        c[12] = 0.995 * y[9] + 60.8 * x[1] + 48 * x[3] - 0.1121 * y[13] - 5095
        y[14] = y[12] / c[12]
        y[15] = 148000 - 331000 * y[14] + 40 * y[12] - 61 * y[14] * y[12]
        c[13] = 2324 * y[9] - 28740000 * y[1]
        c[14] = y[12] * (1 / y[14] - 1 / 0.52)
        c[15] = 1.104 - 0.72 * y[14]
        y[16] = 14130000 - 1328 * y[9] - 531 * y[10] + c[13] / c[11]
        c[16] = y[8] + x[4]
        return y, c
  
    @classmethod
    def f(cls, x):
        y, c = cls.get_vals(x)
        v = 0.000117 * y[13]
        v += 0.1365
        v += 0.00002358 * y[12]
        v += 0.000001502 * y[15]
        v += 0.0321 * y[11]
        v += 0.004324 * y[4]
        v += 1e-4 * c[14] / c[15]
        v += 37.48 * y[1] / c[11]
        v += -0.0000005843 * y[16]
        return v

    @classmethod
    def g(cls, x):
        y, c = cls.get_vals(x)
        g_ = []
        g_.append(0.28 / 0.72 * y[4] - y[3])
        g_.append(x[2] - 1.5 * x[1])
        g_.append(3496 * y[1] / c[11] - 21)
        g_.append(110.6 + y[0] - 62212 / c[16])
        g_.append(213.1 - y[0])
        g_.append(y[0] - 405.23)
        g_.append(17.505 - y[1])
        g_.append(y[1] - 1053.6667)
        g_.append(11.275 - y[2])
        g_.append(y[2] - 35.03)
        g_.append(214.228 - y[3])
        g_.append(y[3] - 665.585)
        g_.append(7.458 - y[4])
        g_.append(y[4] - 584.463)
        g_.append(0.961 - y[5])
        g_.append(y[6] - 265.916)
        g_.append(1.612 - y[6])
        g_.append(y[6] - 7.046)
        g_.append(0.146 - y[7])
        g_.append(y[7] - 0.222)
        g_.append(107.99 - y[8])
        g_.append(y[8] - 273.366)
        g_.append(922.693 - y[9])
        g_.append(y[9] - 1286.105)
        g_.append(926.832 - y[10])
        g_.append(y[10] - 1444.046)
        g_.append(18.766 - y[11])
        g_.append(y[11] - 537.141)
        g_.append(1072.163 - y[12])# la
        g_.append(y[12] - 3247.039)
        g_.append(8961.448 - y[13])
        g_.append(y[13] - 26844.086)
        g_.append(0.063 - y[14])
        g_.append(y[14] - 0.386)
        g_.append(71084.33 - y[15])
        g_.append(y[15] - 140000)
        g_.append(2802713 - y[16])
        g_.append(y[16] - 12146108)
        return np.asarray(g_)


class G17(ConstrainedTestProblem):
    """
    TODO: check
    ERRATUM: CEC def, x[0] < 400, should be <=, according to bounds
    """
    
    dim = 6
    is_ineq = [False] * 4
    bounds = np.asarray([[0, 400], [0, 1000]] + [[340, 420]] * 2  + [[-1000, 1000], [0, 0.5236]]).T
    xopt = (201.784467214523659, 99.9999999999999005, 383.071034852773266, 
            420, -10.9076584514292652, 0.0731482312084287128)
    fmin = 8853.53967480648

    @staticmethod
    def f(x):
        if 0 <= x[0] < 300:
            f1 = 30 * x[0]
        elif 300 <= x[0] <= 400:
            f1 = 31 * x[0]
        if 0 <= x[1] < 100:
            f2 = 28 * x[1]
        elif 100 <= x[1] < 200:
            f2 = 29 * x[1]
        elif 200 <= x[1] <= 1000:
            f2 = 30 * x[1]
        return f1 + f2
    
    @staticmethod
    def g(x):
        g_ = []
        g_.append(-x[0] + 300 - (x[2] * x[3] / 131.078 * np.cos(1.48477 - x[5])) + 0.90798 * x[2]**2 / 131.078 * np.cos(1.47588))
        g_.append(-x[1] - (x[2] * x[3] / 131.078 * np.cos(1.48477 + x[5])) + 0.90798 * x[3]**2 / 131.078 * np.cos(1.47588))
        g_.append(-x[4] - (x[2] * x[3] / 131.078 * np.sin(1.48477 + x[5])) + 0.90798 * x[3]**2 / 131.078 * np.sin(1.47588))
        g_.append(200 - (x[2] * x[3] / 131.078 * np.sin(1.48477 - x[5])) + 0.90798 * x[2]**2 / 131.078 * np.sin(1.47588))
        return np.asarray(g_)


class G18(ConstrainedTestProblem):

    dim = 9
    is_ineq = [True] * 13
    bounds = np.asarray([[-10, 10]] * 8 + [[0, 20]])
    xopt = (-0.657776192427943163, -0.153418773482438542, 0.323413871675240938,
             -0.946257611651304398, -0.657776194376798906, -0.753213434632691414,
             0.323413874123576972, -0.346462947962331735, 0.59979466285217542)
    fmin = -0.866025403784439

    @staticmethod
    def f(x):
        v = x[0] * x[3]
        v += -x[1] * x[2]
        v += x[2] * x[8]
        v += -x[4] * x[8]
        v += x[4] * x[7]
        v += -x[5] * x[6]
        return -0.5 * v

    @staticmethod
    def g(x):
        g_ = []
        g_.append(x[2]**2 + x[3]**2 - 1)
        g_.append(x[8]**2 - 1)
        g_.append(x[4]**2 + x[5]**2 - 1)
        g_.append((x[0] - x[4])**2 + (x[1] - x[5])**2 - 1)
        g_.append((x[0] - x[6])**2 + (x[1] - x[7])**2 - 1)
        g_.append((x[2] - x[4])**2 + (x[3] - x[5])**2 - 1)
        g_.append((x[2] - x[6])**2 + (x[3] - x[7])**2 - 1)
        g_.append(x[6]**2 + (x[7] - x[8])**2 - 1)
        g_.append(x[1] * x[2] - x[0] * x[3])
        g_.append(-x[2] * x[8])
        g_.append(x[4] * x[8])
        g_.append(x[5] * x[6] - x[4] * x[7])
        return np.asarray(g_)
    

class G19(ConstrainedTestProblem):

    dim = 15
    is_ineq = [True] * 5
    bounds = np.asarray([[0, 10]] * 15)
    # test this xopt
    xopt = (0, 0, 3.94599045143233784, 0, 3.2831773458454161, 
             10, 0, 0, 0, 0, 0.370764847417013987, 0.278456024942955571,
             0.523838487672241171, 0.388620152510322781, 0.298156764974678579)
    fmin = 32.6555929502463

    # problem data
    a = [[-16, 2, 0, 1, 0],
         [0, -2, 0, 0.4, 2],
         [-3.5, 0, 2, 0, 0],
         [0, -2, 0, -4, -1],
         [0, -9, -2, 1, -2.8],
         [2, 0, -4, 0, 0],
         [-1] * 5,
         [-1, -2, -3, -2, -1],
         [1, 2, 3, 4, 5],
         [1] * 5]
    a = np.asarray(a)
    b = np.array([-40, -2, -0.25, -4, -4, -1, -40, -60, 5, 1])
    c = [[30, -20, -10, 32, -10],
         [-20, 39, -6, -31, 32],
         [-10, -6, 10, -6, -10],
         [32, -31, -6, 39, -20],
         [-10, 32, -10, -20, 30]]
    c = np.asarray(c)
    d = np.array([4, 8, 10, 6, 2])
    e = np.array([-15, -27, -36, -18, - 12])
    
    @classmethod
    def f(cls, x):
        x = np.asarray(x)
        v = np.dot(np.dot(x[10:], cls.c), x[10:])
        v += 2 * sum(np.multiply(cls.d, x[10:]**3))
        v += - sum(np.multiply(cls.b, x[:10]))
        return v

    @classmethod
    def g(cls, x):
        x = np.asarray(x)
        g_vec = -2 * np.dot(x[10:], cls.c)
        g_vec += -3 * np.multiply(x[10:], cls.d)
        g_vec += -cls.e
        g_vec += np.dot(x[:10], cls.a)
        return g_vec


class G20(ConstrainedTestProblem):
    """ The solution is a little infeasible and no feasible solution 
    has been found so far [cec2006]
    What is meant by a little?
    """

    dim = 24
    is_ineq = [True] * 6 + [False] * 14
    bounds = np.asarray([[0, 10]] * 24)
    # test this xopt
    xopt =(1e-30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.158143376337580827, 0, 0, 0,
            0.530902525044209539, 0, 0,0, 0, 0.310999974151577319,
            5.41244666317833561e-05, 0)
    fmin = None

    # problem data
    a = np.array([0.0693, 0.0577, 0.05, 0.2, 0.26, 0.55, 0.06, 0.1, 0.12,
                  0.18, 0.1, 0.09, 0.0693, 0.0577, 0.05, 0.2, 0.26, 0.55,
                  0.06, 0.1, 0.12, 0.18, 0.1, 0.9])
    b = np.array([44.094, 58.12, 58.12, 137.4, 120.9, 170.9, 62.501, 84.94,
                  133.425, 82.507, 46.07, 60.097, 44.094, 58.12, 58.12, 137.4,
                  120.9, 170.9, 62.501, 84.94, 133.425, 82.507, 46.07, 60.097])
    c = np.array([123.7, 31.7, 45.7, 14.7, 84.7, 27.7, 49.7, 7.1, 2.1, 17.7,
                  0.85, 0.64])
    d = np.array([31.244, 36.12, 34.784, 92.7, 82.7, 91.6, 56.708, 82.7, 80.8,
                  64.517, 49.4, 49.1])
    e = np.array([0.1, 0.3, 0.4, 0.3, 0.6, 0.3])
    k = 0.7302 * 530 * 14.7 * 40
    
    @classmethod
    def f(cls, x):
        x = np.asarray(x)
        return sum(np.multiply(x, cls.a))

    @classmethod
    def g(cls, x):
        x = np.asarray(x)
        g_vec = (x[:3] + x[12:15]) / (sum(x) + cls.e[:3])
        g_vec = np.append(g_vec, (x[3:6] + x[15:18]) / (sum(x) +cls.e[3:6]))
        tmp1 = sum(x[12:] / cls.b[12:])
        tmp2 = 40 * sum(x[:12] / cls.b[:12])
        g_vec = np.append(g_vec, x[12:] / (tmp1 * cls.b[12:])
                          - np.multiply(cls.c, x[:12]) / (tmp2 * cls.b[:12]))
        g_vec = np.append(g_vec, sum(x) - 1)
        g_vec = np.append(g_vec, sum(x[:12] / cls.d)
                          + cls.k * sum(x[12:] / cls.b[12:]) - 1.671)
        return g_vec
    
class G21(ConstrainedTestProblem):
    
    dim = 7
    is_ineq = [True] + [False] * 5
    bounds = np.asarray([[0, 1000]] + [[0, 40]] * 2 + [[100, 300]] 
                        + [[6.3, 6.7]] + [[5.9, 6.4]] + [[4.5, 6.25]]).T
    xopt = (193.724510070034967, 0, 17.3191887294084914, 100.047897801386839,
             6.68445185362377892, 5.99168428444264833, 6.21451648886070451)
    fmin = 193.724510070035
    
    @staticmethod
    def f(x):
        return x[0]
    
    @staticmethod
    def g(x):
        x = np.asarray(x)
        g_ = []
        g_.append(-x[0] + 35 * sum(x[1:2]**0.6))
        g_.append(-300 * x[2] + 7500 * x[4] - 7500 * x[5] - 25 * x[3] * x[4]
                  + 25 * x[3] * x[5] + x[2] * x[3])
        g_.append(100 * x[1] + 155.365 * x[3] + 2500 * x[6] - x[1] * x[3]
                  - 25 * x[3] * x[6] - 15536.5)
        g_.append(-x[4] + np.log(-x[3] + 900))
        g_.append(-x[5] + np.log(x[3] + 300))
        g_.append(-x[6] + np.log(-2 * x[3] + 700))
        return np.asarray(g_)


class G22(ConstrainedTestProblem):
    
    dim = 22
    is_ineq = [True] + [False] * 19
    bounds = np.asarray([[0, 2e4]] + [[0, 1e6]] * 3 + [[0, 4e7]] * 3
                        + [[100, 299.99]] + [[100, 399.99]] + [[100.01, 300]]
                        + [[100, 400]] + [[100, 600]] + [[0, 500]] * 3
                        + [[0.01, 300]] + [[0.01, 400]] + [[-4.7, 6.25]] * 5).T
    xopt = (236.430975504001054, 135.82847151732463, 204.818152544824585, 6446.54654059436416,
             3007540.83940215595, 4074188.65771341929, 32918270.5028952882, 130.075408394314167,
             170.817294970528621, 299.924591605478554, 399.258113423595205, 330.817294971142758,
             184.51831230897065, 248.64670239647424, 127.658546694545862, 269.182627528746707,
             160.000016724090955, 5.29788288102680571, 5.13529735903945728, 5.59531526444068827,
             5.43444479314453499, 5.07517453535834395)
    fmin = 236.430975504001
    
    @staticmethod
    def f(x):
        return x[0]
    
    @staticmethod
    def g(x):
        x = np.asarray(x)
        g_ = []
        g_.append(-x[0] + 35 * sum(x[1:3]**0.6))
        g_.append(x[4] - 1e5 * x[7 + 1e7])
        g_.append(x[5] + 1e5 * x[7] - 1e5 * x[8])
        g_.append(x[6] + 1e5 * x[8] -5e7)
        g_.append(x[4] + 1e5 * x[9] - 3.3e7)
        g_.append(x[5] + 1e5 * x[10] - 4.4e7)
        g_.append(x[6] + 1e5 * x[11] - 6.6e7)
        g_.append(x[4] - 120 * x[1] * x[12])
        g_.append(x[5] - 80 * x[2] * x[13])
        g_.append(x[6] - 40 * x[3] * x[13])
        g_.append(x[7] - x[10] + x[15])
        g_.append(x[8] - x[11] + x[16])
        g_.append(-x[17] + np.log(x[9] - 100))
        g_.append(-x[18] + np.log(-x[7] + 300))
        g_.append(-x[19] + np.log(x[15]))
        g_.append(-x[20] + np.log(-x[8] + 400))
        g_.append(-x[21] + np.log(x[16]))
        g_.append(-x[7] - x[9] + x[12] * x[17] - x[12] * x[18] + 400)
        g_.append(x[7] - x[8] - x[10] + x[13] * (x[19] - x[20]) +400)
        g_.append(x[8] - x[11] + (-4.60517 + x[21]) * x[14] + 100)
        return np.asarray(g_)


class G23(ConstrainedTestProblem):
    
    dim = 9
    is_ineq = [True] * 2+ [False] * 4
    bounds = np.asarray([[0, 1000]] + [[0, 40]] * 2 + [[100, 300]] 
                        + [[6.3, 6.7]] + [[5.9, 6.4]] + [[4.5, 6.25]]).T
    xopt = (0.00510000000000259465, 99.9947000000000514,
            9.01920162996045897e-18, 99.9999000000000535, 0.000100000000027086086, 
            2.75700683389584542e-14, 99.9999999999999574, 2000.0100000100000100008)
    fmin = -400.055099999999584
    
    @staticmethod
    def f(x):
        v = -9 * x[4]
        v += -15 * x[7]
        v += 6 * x[0]
        v += 16 * x[1]
        v += 10 * (x[5] + x[6])
        return v
    
    @staticmethod
    def g(x):
        x = np.asarray(x)
        g_ = []
        g_.append(x[8] * x[2] + 0.02 * x[5] - 0.025 * x[4])
        g_.append(x[8] * x[3] + 0.02 * x[6] - 0.015 * x[7])
        g_.append(x[0] + x[1] - x[2] - x[3])
        g_.append(0.03 * x[0] + 0.01 * x[1] - x[8] * (x[2] + x[3]))
        g_.append(x[2] + x[6] - x[5])
        g_.append(x[3] + x[7] - x[8])
        return np.asarray(g_)


class G24(ConstrainedTestProblem):
    """
    This problem has a feasible
    region consisting on two disconnected sub-regions
    """
    
    dim = 2
    is_ineq = [True] * 2
    bounds = np.asarray([[0, 3], [0, 4]]).T
    xopt = (2.32952019747762, 3.17849307411774)
    fmin = -5.50801327159536
    
    @staticmethod
    def f(x):
        return - x[0] - x[1]
    
    @staticmethod
    def g(x):
        x = np.asarray(x)
        g_ = []
        g_.append(-2 * x[0]**4 + 8 * x[0]**3 - 8 * x[0]**2 + x[1] - 2)
        g_.append(-4 * x[0]**4 + 32 * x[0]**3 - 88 * x[0]**2 + 96 * x[0] + x[1] - 36)
        return np.asarray(g_)


class PB240(ConstrainedTestProblem):
    """
    >>> PB240.f(PB240.x_start)
    -1250
    >>> PB240.g(PB240.x_start)
    array([-35000])
    """

    is_ineq = [True]
    bounds = np.asarray([[0, np.inf]] * 5).T
    dim = 5
    fmin = -5000
    x_start = [250] * 5

    @staticmethod
    def f(x):
        return - sum(x)

    @staticmethod
    def g(x):
        return np.asarray([sum(np.multiply(9 + np.arange(1, len(x)+1), x)) - 50000])

    def g_bounds(self, x):
        return - np.asarray(x)

class PB241(ConstrainedTestProblem):
    """
    >>> PB241.f(PB241.x_start)
    -3750
    """

    is_ineq = [True]
    bounds = np.asarray([[0, np.inf]] * 5).T
    dim = 5
    fmin = -125000/7
    x_start = [250] * 5

    @staticmethod
    def f(x):
        return - sum(np.multiply(np.arange(1, len(x)+1), x))

    @staticmethod
    def g(x):
        return np.asarray([sum(np.multiply(9 + np.arange(1, len(x)+1), x)) - 50000])
    
    def g_bounds(self, x):
        return - np.asarray(x)


class TR2(ConstrainedTestProblem):
    
    is_ineq = [True]
    bounds = None
    dim = 2
    fmin = 2
    x_start = [50] * 2
    
    @staticmethod
    def f(x):
        x = np.asarray(x)
        return sum(x**2)

    @staticmethod
    def g(x):
        return [len(x) - sum(x)]


arnold2012 = {
    "HB": {"obj": G4}, 
    "G6": {"obj": G6},
    "G7": {"obj": G7}, 
    "G9": {"obj": G9}, 
    "G10": {"obj": G10},
    "2.40": {"obj": PB240},
    "2.41": {"obj": PB241},
    "TR2": {"obj": TR2}
}

cec2006 = {
    "G1": G1,
    "G2": G2,
    "G3": G3,
    "G4": G4,
    "G5": G5,
    "G6": G6,
    "G7": G7,
    "G8": G8,
    "G9": G9,
    "G10": G10,
    "G11": G11,
    "G12": G12,
    "G13": G13,
    "G14": G14,
    "G15": G15,
    "G16": G16,
    "G17": G17,
    "G18": G18,
    "G19": G19,
    "G20": G20,
    "G21": G21,
    "G22": G22,
    "G23": G23,
    "G24": G24,
    }

# Constructor for artificial test problems


def sphere(x):
    # Sphere function, separable, f(0) = 0
    return sum(x**2)


def elli(x, cond=1e6):
    # Ellipsoid functionn, separable, f(0) = 0
    N = len(x)
    return sum(cond**(np.arange(N) / (N - 1.)) * x**2)


class TranslatedFunction:
    """
    Utility to shift function argmin location by 1
    """

    def __init__(self, func, coord):
        self.func = func
        self.coord = coord

    def __call__(self, x):
        y = np.array(x) # NB: array not asarray here to copy
        y[self.coord] -= 1
        return self.func(y)


class LinConsQP(ConstrainedTestProblem):
    """
    Quadratic problem with a linear constraints. 
    Can also implement a non linear constraint equivalent to x < 0
    For research purpose
    >>> problem = LinConsQP()
    >>> problem(np.zeros(10))
    (1.0, array([0.]))
    
    We can change m on the fly:
    >>> problem.m = 3; problem(np.zeros(10))
    (3.0, array([0., 0., 0.]))
    """

    is_ineq = None
    dim = None
    m = None

    def __init__(self, seed=1, eps_feas=0, dim=10, m=1, obj="sphere", cons="lin"):
        """
        Instantiate a linearly constrained quadratic problem

        Parameters
        ----------
        seed : int
            Random seed passed to base class
        dim : int
            Problem dimension
        m : int
            Number of linear constraints
        obj : string or callable
            Objective function
            if string: sphere or elli
            if callable: user-specified quadratic function
        cons : string or callable
            Constraints function
            if string: lin  for linear
            if callable: user-specified function that preserves R+, R-
        """
        super().__init__(seed, eps_feas)
        self.dim = dim
        self.m = m
        # Objective function construction
        if isinstance(obj, str):
            if obj == "sphere":
                self._f = sphere
            if obj == "elli":
                self._f = elli
        elif callable(obj):
            self._f = obj
        else:
            raise ValueError
        # Constraints function construction
        self.is_ineq = [True] * m
        if isinstance(cons, str):
            if cons == "lin":
                self._g = lambda x: [x[i] for i in range(self.m)]
        elif callable(cons):
            self._g = cons
        else:
            raise ValueError
        try:
            assert len(self.g(np.ones(self.dim))) == self.m
        except AssertionError:
            print(f"m={self.m}")
            print(f"g={self.g(np.ones(self.dim))}")
            raise AssertionError

        self.fmin = m
        self.xopt = np.zeros(dim)

    @property
    def coord(self):
        return np.arange(self.m)
    
    def f(self, x):
        y = np.array(x)
        y[self.coord] -= 1
        return self._f(y)

    def g(self, x):
        return np.asarray(self._g(x), dtype=float)

class NoisyLinConsQP(LinConsQP):

    def __init__(self, *args, **kw):
        self.sigma_f = 0
        self.sigma_g = 0
        super().__init__(*args, **kw)

    def set_noise(self, sigma_f, sigma_g):
        self.sigma_f = sigma_f
        self.sigma_g = sigma_g

    def f(self, x):
        f_ = super().f(x)
        if self.sigma_f:
            f_ += np.random.normal(0, self.sigma_f)
        return f_

    def g(self, x):
        g_ = super().g(x)
        if self.sigma_g:
            #print(g_)
            g_ += np.random.normal(0, self.sigma_g)
        return g_


if __name__ == "__main__":
    import doctest
    doctest.testmod()

