import numpy as np
import matplotlib.pyplot as plt

"""
粒子群算法实现步骤：
(1)初始化：
     初始化参数：学习因子c1,c2:一般取2; 惯性权重w:一般取0.9-1.2; 种群数量N：50-1000;迭代次数：100-4000;空间维数：自变量的个数;位置限制：自变量的取值范围;速度限制：可以控制在0-1之间。
     初始化粒子群：包括粒子群的初始位置和速度
     根据初始化得到的粒子群计算p_best,g_best
(2)更新粒子位置和粒子的速度
(3)计算每个粒子的适应度值
(4)根据适应度值更新p_best(个体最佳历史位置),g_best(全局最佳历史位置)
(5)判断是否满足结束条件，满足输出最优解，不满足转到第(2)
"""


class PSO(object):
    def __init__(self, population_size, max_steps):
        self.x = None
        self.w = 0.6  # 惯性权重
        self.c1 = self.c2 = 2  # 学习因子
        self.population_size = population_size  # 粒子群数量
        self.dim = 4  # 搜索空间的维度
        self.max_steps = max_steps  # 迭代次数
        self.bound = []  # 自变量的约束范围
        self.bound.append([1, 1, 1, 1])
        self.x = np.zeros((self.population_size, self.dim))
        self.bound.append([30, 30, 30, 30])  # 解空间范围
        for i in range(self.population_size):  # 对于每一只蚂蚁来说都会有一个潜在解
            for j in range(self.dim):  # 取出每一个变量,产生一个在[var_num_min,var_num_max]范围内的随机数
                self.x[i][j] = np.random.uniform(self.bound[0][j], self.bound[1][j])
        self.v = np.random.rand(self.population_size, self.dim)  # 初始化粒子群速度
        # print("v", self.v)  # population行4列，
        # fitness = self.calculate_fitness(self.x)
        self.Fitness = []
        for i in range(self.population_size):
            fitness = self.calculate_fitness(self.x[i])
            self.Fitness.append(fitness)
        self.p_best = self.x  # 个体的最优位置
        print("个体最优位置：", self.p_best)
        self.g_best = self.x[np.argmax(self.Fitness)]  # 全局最优位置
        print("全局最优位置：", self.g_best)
        self.individual_best_fitness = self.Fitness  # 个体的最优适应度
        print("个体最优适应度：", self.individual_best_fitness)
        self.global_best_fitness = np.max(self.Fitness)  # 全局最佳适应度
        print("全局最优适应度：", self.global_best_fitness)

    def calculate_fitness(self, x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        y = x1 ** 2 + x2 ** 2 + x3 ** 3 + x4 ** 4
        # print("fitness:", y)
        return y

    def evolve(self):  # evolve进化的意思
        fig = plt.figure()
        for step in range(self.max_steps):
            r1 = np.random.rand(self.population_size, self.dim)  # 返回一个或者一组0-1之间的随机数或者随机数组
            r2 = np.random.rand(self.population_size, self.dim)
            # 更新速度和权重
            #
            self.v = self.w * self.v + self.c1 * r1 * (self.p_best - self.x) + self.c2 * r2 * (self.g_best - self.x)
            self.x = self.v + self.x
            for i in range(self.population_size):
                for j in range(self.dim):
                    if self.v[i][j] < self.bound[0][j]:
                        self.v[i][j] = self.bound[0][j]
                    if self.x[i][j] > self.bound[1][j]:
                        self.x[i][j] = self.bound[1][j]

            for i in range(self.population_size):  # 对于每一个粒子.
                fitness = self.calculate_fitness(self.x[i])  # 新的粒子群的适应度，100行1列
                # 需要更新的个体
                if fitness > self.individual_best_fitness[i]:
                    self.p_best[i] = self.x[i]
                    self.individual_best_fitness[i] = fitness  # ppt中没有说要更新个体历史适应度，但我觉得应该是要更新的，这样收敛的速度会加快
                    print("用当前位置更新粒子个体的历史最优位置p_best")
                    print("更改之后的粒子最优位置：", self.p_best[i])
                else:
                    print("当前个体适应度值 小于 这个个体的历史最优位置的适应度值，不用更新")
                if fitness > self.global_best_fitness:
                    self.g_best = self.x[i]
                    self.global_best_fitness = fitness
            print("个体最优适应度：", self.individual_best_fitness)
            print('best fitness : %.5f,mean fitness : %.5f' % (self.global_best_fitness, np.mean(fitness)))


pso = PSO(10, 100)
pso.evolve()

