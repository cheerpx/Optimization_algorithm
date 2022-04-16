import numpy as np
import matplotlib.pyplot as plt

"""
蚁群算法流程：
(1)初始化：
    初始化参数：alpha:信息素启发因子([0-5]); beta:期望值启发因子([0-5);rou:信息素挥发因子([0.1-0.99]);种群大小population_size[10,100]
    初始化蚂蚁种群：种群的大小为N，则初始化数组的大小就是（N，），如果是普通的函数优化的话，列数是自变量的个数
    初始化信息素：第一代信息素采用第一代蚁群的适应度值表示
(2)状态转移：根据上一代蚂蚁的信息素和距离等信息计算状态转移概率，根据状态转移概率进行局部搜索或者全局搜索
(3)约束边界：也称为越界保护
(4)选择：产生新种群的操作，根据目标函数值在原始蚁群和状态转移后的蚁群之间进行对比选择，如果原始蚁群的适应度更好，就保留，否则就转移
(5)更新信息素：有具体的公式
(6)对选择后的蚁群重复进行状态转移、约束边界和更新信息素3步，一直到结束。

"""


class ACO:
    def __init__(self, parameters):
        """
        Ant Colony Optimization
        :param parameters: a list type,Like[iter,population_size,var_num_min,var_num_max]
        """
        # 初始化
        # 初始化方法接受传入的参数，包括最大值，最小值，种群大小和迭代代数。
        # 通过这些参数初始化产生一个蚂蚁种群，并记录当前代全局最优位置。
        self.iter = parameters[0]  # 迭代的次数
        self.population_size = parameters[1]  # 种群的大小
        self.var_num = len(parameters[2])  # 自变量的个数
        self.bound = []  # 自变量的约束范围
        self.bound.append(parameters[2])
        self.bound.append(parameters[3])
        self.population_x = np.zeros((self.population_size, self.var_num))  # 所有蚂蚁的位置
        self.g_best = np.zeros((1, self.var_num))  # 全局蚂蚁最优的位置
        # 初始化第0代 初始的全局最优解
        temp = -1
        for i in range(self.population_size):  # 对于每一只蚂蚁来说都会有一个潜在解
            for j in range(self.var_num):  # 取出每一个变量,产生一个在[var_num_min,var_num_max]范围内的随机数
                self.population_x[i][j] = np.random.uniform(self.bound[0][j], self.bound[1][j])
            fit = self.fitness(self.population_x[i])
            if fit > temp:
                self.g_best = self.population_x[i]
                temp = fit
        print("初始种群：", self.population_x)

    def fitness(self, ind_var):
        """
        个体适应值计算
        """
        x1 = ind_var[0]
        x2 = ind_var[1]
        x3 = ind_var[2]
        x4 = ind_var[3]
        y = x1 ** 2 + x2 ** 2 + x3 ** 3 + x4 ** 4
        return y

    # 更新位置信息和信息素信息
    def update_operator(self, gen, tau, tau_max):
        rou = 0.8  # 信息素挥发系数
        Q = 1  # 每只蚂蚁的信息素释放总量
        lamda = 1 / gen
        pi = np.zeros(self.population_size)
        for i in range(self.population_size):  # 对于每一只蚂蚁
            for j in range(self.var_num):  # 对于每一个变量
                pi[i] = (tau_max - tau[i]) / tau_max  # 计算蚂蚁 i 的状态转移概率
                # 更新位置
                if pi[i] < np.random.uniform(0, 1):  # 这里也可以直接使用一个常数来判断是进行局部搜索还是全局搜索
                    # 局部搜索公式：new = old + r1 *step* lamda
                    self.population_x[i][j] = self.population_x[i][j] + np.random.uniform(-1, 1) * lamda
                else:
                    # 全局搜索公式：new = old + r2 *range
                    self.population_x[i][j] = self.population_x[i][j] + np.random.uniform(-1, 1) * (
                            self.bound[1][j] - self.bound[0][j]) / 2
                # 越界保护
                if self.population_x[i][j] < self.bound[0][j]:
                    self.population_x[i][j] = self.bound[0][j]
                if self.population_x[i][j] > self.bound[1][j]:
                    self.population_x[i][j] = self.bound[1][j]


            # 更新每只蚂蚁的信息素
            tau[i] = (1 - rou) * tau[i] + Q * self.fitness(self.population_x[i])  # 蚁密模型

            # 更新全局最优值
            if self.fitness(self.population_x[i]) > self.fitness(self.g_best):
                self.g_best = self.population_x[i]

        tau_max = np.max(tau)

        return tau_max, tau  # t_max是信息素最大的那个值，t存放着每一只蚂蚁在一代中的信息素

    # 主程序
    def main(self):
        popobj = []
        best = np.zeros((1, self.var_num))[0]
        for gen in range(1, self.iter + 1):
            # 第一代的时候，初始的信息素根据第一代的适应度值计算
            if gen == 1:
                tau_init = np.array(list(map(self.fitness, self.population_x)))
                # map()是Python内置的高阶函数，它接受一个函数f 和一个list,并通过把函数f 依次作用在list的每一个元素上，得到一个新的list并返回，map函数一定是要经过list()转换的
                tau_max, tau = self.update_operator(gen, tau_init, np.max(tau_init))
            else:
                # 状态转移：更新位置与信息素
                tau_max, tau = self.update_operator(gen, tau, tau_max)
            popobj.append(self.fitness(self.g_best))
            print('############ Generation {} ############'.format(str(gen)))
            print(self.g_best)
            print(self.fitness(self.g_best))
            if self.fitness(self.g_best) > self.fitness(best):
                best = self.g_best.copy()
            print('最好的位置：{}'.format(best))
            print('最大的函数值：{}'.format(self.fitness(best)))
        print("---- End of (successful) Searching ----")

        plt.figure()
        plt.title("Figure1")
        plt.xlabel("iterators", size=14)
        plt.ylabel("fitness", size=14)
        t = [t for t in range(1, self.iter + 1)]
        plt.plot(t, popobj, color='b', linewidth=2)
        plt.show()


if __name__ == '__main__':
    iter = 100
    population_size = 100
    low = [1, 1, 1, 1]
    up = [30, 30, 30, 30]
    parameters = [iter, population_size, low, up]
    aco = ACO(parameters)
    aco.main()
