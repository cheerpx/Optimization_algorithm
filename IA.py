import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
G = 100  # 迭代次数
fun = '5 * sin(x * y) + x ** 2 + y ** 2'  # 目标函数
choice = 'max'
dim = 2  # 维度
limit_x = [-4, 4]  # x取值范围
limit_y = [-4, 4]  # y取值范围
pop_size = 10  # 种群规模
mutate_rate = 0.7  # 变异概率
delta = 0.2  # 相似度阈值
beta = 1  # 激励度系数
colone_num = 10  # 克隆份数
if choice == 'max':
    alpha = 2  # 激励度系数,求最大值为正,最小值为负
else:
    alpha = -2

def func(x, y):
    fx = 5 * np.sin(x * y) + x ** 2 + y ** 2
    return fx



# 初始化种群
def init_pop(dim, pop_size, *limit):
    pop = np.random.rand(dim, pop_size)
    for i in range(dim):
        pop[i, :] *= (limit[i][1] - limit[i][0])
        pop[i, :] += limit[i][0]

    return pop


# 计算浓度
def calc_density(pop, delta):
    density = np.zeros([pop.shape[1], ])
    for i in range(pop.shape[1]):
        density[i] = np.sum(
            len(np.ones([pop.shape[1], ])[np.sqrt(np.sum((pop - pop[:, i].reshape([2, 1])) ** 2, axis=0)) < delta]))
    return density / pop.shape[1]


# 计算激励度
def calc_simulation(simulation, density):
    return (alpha * simulation - beta * density)


# 变异，随着代数增加变异范围逐渐减小
def mutate(x, gen, mutata_rate, dim, *limit):
    for i in range(dim):
        if np.random.rand() <= mutata_rate:
            x[i] += (np.random.rand() - 0.5) * (limit[i][1] - limit[i][0]) / (gen + 1)  # 加上随机数产生变异
            if (x[i] > limit[i][1]) or (x[i] < limit[i][0]):  # 边界检测
                x[i] = np.random.rand() * (limit[i][1] - limit[i][0]) + limit[i][0]


pop = init_pop(dim, pop_size, limit_x, limit_y)
init_simulation = func(pop[0, :], pop[1, :])
density = calc_density(pop, delta)
simulation = calc_simulation(init_simulation, density)
# print(pop)
# print(init_simulation)
# print(calc_density(pop,delta))
# 进行激励度排序
index = np.argsort(-simulation)
pop = pop[:, index]
new_pop = pop.copy()  # 浅拷贝
simulation = simulation[index]

# 免疫循环
for gen in range(G):
    best_a = np.zeros([dim, int(pop_size / 2)])  # 用于保留每次克隆后亲和度最高的个体
    best_a_simulation = np.zeros([int(pop_size / 2), ])  # 保存激励度
    # 选出激励度前50%的个体进行免疫
    for i in range(int(pop_size / 2)):
        a = new_pop[:, i].reshape([2, 1])
        b = np.tile(a, (1, colone_num))  # 克隆10份
        for j in range(colone_num):
            mutate(a, gen, mutate_rate, dim, limit_x, limit_y)
        b[:, 0] = pop[:, i]  # 保留克隆源个体
        # 保留亲和度最高的个体
        b_simulation = func(b[0, :], b[1, :])
        index = np.argsort(-b_simulation)
        best_a_simulation = b_simulation[index][0]  # 最佳个体亲和度
        best_a[:, i] = b[:, index][:, 0]  # 最佳个体
    # 计算免疫种群的激励度
    a_density = calc_density(best_a, delta)
    best_a_simulation = calc_simulation(best_a_simulation, a_density)
    # 种群刷新
    new_a = init_pop(2, int(pop_size / 2), limit_x, limit_y)
    # 新生种群激励度
    new_a_simulation = func(new_a[0, :], new_a[1, :])
    new_a_density = calc_density(new_a, delta)
    new_a_simulation = calc_simulation(new_a_simulation, new_a_density)
    # 免疫种群与新生种群合并
    pop = np.hstack([best_a, new_a])
    simulation = np.hstack([best_a_simulation, new_a_simulation])
    index = np.argsort(-simulation)
    pop = pop[:, index]
    simulation = simulation[index]
    new_pop = pop.copy()

# 新建一个画布
figure = plt.figure(figsize=(10, 8), dpi=80)
# 新建一个3d绘图对象
ax = Axes3D(figure)
# 定义x,y 轴名称
plt.xlabel("x")
plt.ylabel("y")
for i in range(int(pop_size / 2)):
    ax.scatter(pop[0, i], pop[1, i], func(pop[0, i], pop[1, i]), color='red')
    print('最优解:', 'x = ', pop[0, i], 'y = ', pop[1, i], end='\n')
    print('结果:', 'z = ', func(pop[0, i], pop[1, i]))
x = np.arange(limit_x[0], limit_x[1], (limit_x[1] - limit_x[0]) / 50)
y = np.arange(limit_y[0], limit_y[1], (limit_y[1] - limit_y[0]) / 50)
x, y = np.meshgrid(x, y)
z = func(x, y)
ax.plot_surface(x, y, z, rstride=1, cstride=1, color='green', alpha=0.5)
plt.show()