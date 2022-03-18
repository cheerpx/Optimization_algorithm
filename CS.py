from random import uniform
from random import randint
import math
import numpy as np
import matplotlib.pyplot as plt

'''
根据levy飞行计算新的巢穴位置
'''


def GetNewNestViaLevy(Xt, Xbest, Lb, Ub, lamuda):
    beta = 1.5
    sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (
            math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
    sigma_v = 1
    for i in range(Xt.shape[0]):
        s = Xt[i, :]
        u = np.random.normal(0, sigma_u, 1)
        v = np.random.normal(0, sigma_v, 1)
        Ls = u / ((abs(v)) ** (1 / beta))
        stepsize = lamuda * Ls * (s - Xbest)  # lamuda的设置关系到点的活力程度  方向是由最佳位置确定的  有点类似PSO算法  但是步长不一样
        s = s + stepsize * np.random.randn(1, len(s))  # 产生满足正态分布的序列
        Xt[i, :] = s
        Xt[i, :] = simplebounds(s, Lb, Ub)
    return Xt


'''
按pa抛弃部分巢穴
'''


def empty_nests(nest, Lb, Ub, pa):
    n = nest.shape[0]
    nest1 = nest.copy()
    nest2 = nest.copy()
    rand_m = pa - np.random.rand(n, nest.shape[1])
    rand_m = np.heaviside(rand_m, 0)
    np.random.shuffle(nest1)
    np.random.shuffle(nest2)
    # stepsize = np.random.rand(1,1) * (nest1 - nest)
    stepsize = np.random.rand(1, 1) * (nest1 - nest2)
    new_nest = nest + stepsize * rand_m
    nest = simplebounds(new_nest, Lb, Ub)
    return nest


'''
获得当前最优解
'''


def get_best_nest(nest, newnest, Nbest, nest_best):
    # get_best_nest(nest, nest, Nbest, nest_best)
    fitall = 0
    for i in range(nest.shape[0]):
        temp1 = fitness(nest[i, :])
        temp2 = fitness(newnest[i, :])
        if temp1 > temp2:
            nest[i, :] = newnest[i, :]
            if temp2 < Nbest:
                Nbest = temp2
                nest_best = nest[i, :]
            fitall = fitall + temp2
        else:
            fitall = fitall + temp1
    meanfit = fitall / nest.shape[0]
    return nest, Nbest, nest_best, meanfit


'''
进行适应度计算
'''


def fitness(nest_n):
    X = nest_n[0]
    Y = nest_n[1]
    # rastrigin函数
    A = 10
    Z = 2 * A + X ** 2 - A * np.cos(2 * np.pi * X) + Y ** 2 - A * np.cos(2 * np.pi * Y)

    return Z


'''
进行全部适应度计算
'''


def fit_function(X, Y):
    # rastrigin函数
    A = 10
    Z = 2 * A + X ** 2 - A * np.cos(2 * np.pi * X) + Y ** 2 - A * np.cos(2 * np.pi * Y)
    return Z


'''
约束迭代结果
'''


def simplebounds(s, Lb, Ub):
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            if s[i][j] < Lb[j]:
                s[i][j] = Lb[j]
            if s[i][j] > Ub[j]:
                s[i][j] = Ub[j]
    return s


def Get_CS(lamuda=1, pa=0.25):
    Lb = [-5, -5]  # 下界
    Ub = [5, 5]  # 上界
    population_size = 20
    dim = 2
    nest = np.random.uniform(Lb[0], Ub[0], (population_size, dim))  # 初始化位置
    nest_best = nest[0, :]
    Nbest = fitness(nest_best)
    nest, Nbest, nest_best, fitmean = get_best_nest(nest, nest, Nbest, nest_best)
    for i in range(30):
        nest_c = nest.copy()
        newnest = GetNewNestViaLevy(nest_c, nest_best, Lb, Ub, lamuda)  # 根据莱维飞行产生新的位置

        nest, Nbest, nest_best, fitmean = get_best_nest(nest, newnest, Nbest, nest_best)  # 判断新的位置优劣进行替换

        nest_e = nest.copy()
        newnest = empty_nests(nest_e, Lb, Ub, pa)  # 丢弃部分巢穴

        nest, Nbest, nest_best, fitmean = get_best_nest(nest, newnest, Nbest, nest_best)  # 再次判断新的位置优劣进行替换

    print("最优解的适应度函数值", Nbest)
    return Nbest


Get_CS()