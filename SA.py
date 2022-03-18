import math
import matplotlib.pyplot as plt
from random import random


#(1)初始化：初始温度、初始新解
#（2） 添加随机扰动，产生新解
#（3）比较能量差，判断新解是否接受
#（4）如果接受，并且迭代没有结束，继续产生新解，重复（2）（3）
#（5）判断是否达到终止条件，如果没有，降低温度，重新产生新解。。。。如此循环



#首先定义目标函数
def func(x,y):
    res = 4 * x ** 2 - 2.1 * x ** 4 + x ** 6 / 3 + x * y - 4 * y ** 2 + 4 * y ** 4
    return res



class SA:
    def __init__(self, func, iter = 100, T0 = 100, Tf = 0.01, alpha=0.99):
        self.func = func
        self.iter = iter
        self.alpha = alpha
        self.T0 = T0      #退火速率
        self.Tf = Tf      #终止温度
        self.T = T0       #当前温度
        self.x = [random() * 11 - 5 for i in range(iter)]   #随机生成100个x1的值
        self.y = [random() * 11 - 5 for i in range(iter)]   #随机生成100个x2的值
        self.most_best = []
        """
        random()这个函数取0到1之间的小数
        如果你要取0-10之间的整数，包括0和10，就写成(int)random()*11就可以了。11乘以零点多的数最大是10点多，最小是0点多
        该实例中x1和x2的绝对值不超过5，包括-5和5，random() * 11 - 5的结果就是-6到6之间的任意值，不包括-6和6
        所以先乘以11，取-6到6之间的值，产生新解过程中，用一个if条件语句把-5到5之间（包括整数5和-5）的筛选出来。
        """
        self.history = {'f': [], 'T': []}

    def generate_new(self, x, y):   #扰动产生新解的过程
        while True:
            x_new = x + self.T * (random() - random())
            y_new = y + self.T * (random() - random())
            if(-5 <= x_new <= 5) & (-5 <= y_new <= 5):
                break
        return x_new, y_new

    def Metrospolis(self, f, f_new):
        if f_new <= f:
            return 1   #完全接受
        else:
            p = math.exp((f - f_new) / self.T)
            if random() < p:
                return 1
            else:
                return 0

    def best(self):
        f_list = []   #f_list数组保存每次迭代之后的值
        for i in range(self.iter):
            f = self.func(self.x[i], self.y[i])
            f_list.append(f)

        f_best = min(f_list)

        idx = f_list.index(f_best)
        return f_best, idx  #f_best,idx分别是在该温度下，迭代L次之后目标函数的最优解和最优解的下标

    def run(self):
        count = 0
        #外循环迭代，当前温度小于终止温度的阈值
        while self.T > self.Tf:
            #内循环迭代100次
            for i in range(self.iter):
                f = self.func(self.x[i], self.y[i])
                x_new, y_new = self.generate_new(self.x[i], self.y[i])
                f_new = self.func(x_new, y_new)
                if self.Metrospolis(f, f_new):
                    self.x[i] = x_new
                    self.y[i] = y_new

            #迭代L次记录在该温度下的最优解
            ft,_ = self.best()
            self.history['f'].append(ft)
            self.history['T'].append(self.T)

            #温度按照一定的比例下降（冷却）
            self.T = self.T * self.alpha
            count += 1

            #得到最优解

        f_best,idx = self.best()
        print(f"F={f_best},x={self.x[idx]},y={self.y[idx]}")


sa = SA(func)
sa.run()

plt.plot(sa.history['T'],sa.history['f'])
plt.title('SA')
plt.xlabel('T')
plt.ylabel('f')
plt.gca().invert_xaxis()
plt.show()