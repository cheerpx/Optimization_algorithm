import math
import matplotlib.pyplot as plt
import numpy as np
from random import random

# 首先要明确的是 模拟退火算法是一个局部搜索算法，它一般结合全局搜索算法使用，用来打辅助
"""
模拟退火算法步骤：
(1)初始化：初始温度T0；终止条件 T<1 ; 退火速率alpha（0.8-0.99）;每个温度下的迭代次数L
(2)在给定温度下，产生新解，计算增量，按照Metropolis准则判断是否接受新解，如果接受，用新解代替旧解，同时修正目标函数
(3)降低温度T，回到第二步
(4)如果满足终止条件则输出当前解作为最优解，结束程序。否则，回到第（2）步

"""


# 首先定义目标函数
def func(x, y):
    res = 4 * x ** 2 - 2.1 * x ** 4 + x ** 6 / 3 + x * y - 4 * y ** 2 + 4 * y ** 4
    return res


class SA:
    def __init__(self, func, iter=100, T0=100, Tf=0.01, alpha=0.99):
        self.func = func  # 目标函数
        self.iter = iter  # 迭代次数
        self.alpha = alpha  # 退火速率
        self.T0 = T0  # 初始温度
        self.Tf = Tf  # 终止温度
        self.T = T0  # 当前温度
        self.x = [np.random.uniform(-5, 5) for i in range(iter)]  # 随机生成100个x1的值
        self.y = [np.random.uniform(-5, 5) for i in range(iter)]  # 随机生成100个x2的值
        self.most_best = []
        self.history = {'f': [], 'T': []}

    def generate_new(self, x, y):  # 扰动产生新解的过程:通常选择由当前新解经过简单地变换即可产生新解的方法
        while True:  # 这里用的是随机扰动,感觉这个产生新解的方法不是很好，因为新解和旧解之间的差异很大
            # print("旧解：", x)
            x_new = x + self.T * (random() - random())
            # print("新解：", x_new)
            y_new = y + self.T * (random() - random())
            if (-5 <= x_new <= 5) & (-5 <= y_new <= 5):
                break
        return x_new, y_new

    def Metrospolis(self, f, f_new):
        if f_new <= f:
            return 1  # 完全接受
        else:
            p = math.exp((f - f_new) / self.T)
            if random() < p:
                return 1
            else:
                return 0

    def best(self):
        f_list = []  # f_list数组保存每次迭代之后的值
        for i in range(self.iter):
            f = self.func(self.x[i], self.y[i])   # 计算每一代的适应度值
            f_list.append(f)
        f_best = min(f_list)   # 找出100代中适应度值最小的那个值，就是f_best
        idx = f_list.index(f_best)    #
        return f_best, idx  # f_best,idx分别是在该温度下，迭代L次之后目标函数的最优解和最优解的下标

    def run(self):
        # 外循环迭代，当前温度小于终止温度的阈值
        while self.T > self.Tf:
            # 内循环迭代100次
            for i in range(self.iter):
                #print("第", i, "代")
                # 计算解的适应度值
                f = self.func(self.x[i], self.y[i])
                # 给定温度下，产生新解
                x_new, y_new = self.generate_new(self.x[i], self.y[i])
                # 计算新解的适应度值
                f_new = self.func(x_new, y_new)
                # 依据Metropolis原则，进行新解的接受与否判断
                if self.Metrospolis(f, f_new):  # 如果判断是接受新解，就新解代替旧解
                    self.x[i] = x_new
                    self.y[i] = y_new
            #print(self.x)
            # 记录在T温度下的最优解（迭代L次记录在该温度下的最优解）
            ft, _ = self.best()
            self.history['f'].append(ft)
            self.history['T'].append(self.T)
            # 温度按照一定的比例下降（冷却）
            self.T = self.T * self.alpha
            # 得到最优解
        f_best, idx = self.best()
        print(f"F={f_best},x={self.x[idx]},y={self.y[idx]}")


sa = SA(func)
sa.run()
plt.plot(sa.history['T'], sa.history['f'])
plt.title('SA')
plt.xlabel('T')
plt.ylabel('f')
plt.gca().invert_xaxis()
plt.show()
