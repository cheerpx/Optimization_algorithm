import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
免疫优化算法步骤：
（1）初始化：
        初始化参数：抗体种群N(10-100); 最大进化代数(100-500);变异概率
        初始化抗体种群：
（2）计算亲和度：亲和度评价算子
（3）计算抗体浓度和激励度：抗体浓度评价算子，激励度计算算子
（4）免疫选择：免疫选择算子
（5）克隆、变异、克隆抑制：克隆算子、变异算子、克隆抑制算子
（6）种群刷新：种群刷新算子
（7）判断是否满足终止条件，满足输出最优解，不满足回到（2）
"""


class IA:
    def __init__(self, G=100, pop_size=50, mutate_rate=0.7, delta=0.2, beta=1, colone_num=10):
        self.G = G  # 迭代次数
        self.dim = 2  # 自变量个数
        self.limit_x = [-4, 4]  # x取值范围
        self.limit_y = [-4, 4]  # y取值范围
        self.choice = 'max'
        self.pop_size = pop_size  # 种群规模
        self.mutate_rate = mutate_rate  # 变异概率
        self.delta = delta  # 相似度阈值
        self.beta = beta  # 激励度系数
        self.colone_num = colone_num  # colone_num = 10  # 克隆份数
        if self.choice == 'max':
            self.alpha = 2  # 激励度系数,求最大值为正,最小值为负
        else:
            self.alpha = -2

    def func(self, x, y):  # 亲和度计算
        fx = 5 * np.sin(x * y) + x ** 2 + y ** 2
        return fx

    # 初始化种群
    def init_pop(self, dim, pop_size, *limit):
        pop = np.random.rand(dim, pop_size)
        for i in range(dim):
            pop[i, :] *= (limit[i][1] - limit[i][0])
            pop[i, :] += limit[i][0]
        return pop

    # 抗体浓度评价算子：计算浓度
    def calc_density(self, pop, delta):  # 需要一个相似度阈值：delta
        density = np.zeros([pop.shape[1], ])
        for i in range(pop.shape[1]):
            density[i] = np.sum(
                len(np.ones([pop.shape[1], ])[np.sqrt(np.sum((pop - pop[:, i].reshape([2, 1])) ** 2, axis=0)) < delta]))
        return density / pop.shape[1]

    # 激励度算子：计算激励度：计算公式和ppt上的对应
    def calc_simulation(self, simulation, density):
        return (self.alpha * simulation - self.beta * density)

    # 变异，随着代数增加变异范围逐渐减小
    # 变异算子时免疫算法中产生有潜力的新抗体、实现区域搜索的重要算子，它对算法的性能有很大的影响
    # 变异算子也和算子的编码方式相关，实数编码的算法和离散编码的算法采用不同的变异算子：具体的可以看ppt
    def mutate(self, x, gen, mutata_rate, dim, *limit):
        # 这个例子中用的是实数编码变异算子
        for i in range(dim):
            if np.random.rand() <= mutata_rate:
                x[i] += (np.random.rand() - 0.5) * (limit[i][1] - limit[i][0]) / (gen + 1)  # 加上随机数产生变异
                if (x[i] > limit[i][1]) or (x[i] < limit[i][0]):  # 边界检测：边界保护
                    x[i] = np.random.rand() * (limit[i][1] - limit[i][0]) + limit[i][0]

    def run(self):
        # 初始化抗体种群
        pop = self.init_pop(self.dim, self.pop_size, self.limit_x, self.limit_y)
        # print("初始化抗体种群：", pop)  # shape：10行2列
        # 初始化激励度,初始时用适应度函数表示，也称为亲和度计算
        init_simulation = self.func(pop[0, :], pop[1, :])
        # print("初始化激励度：", init_simulation)  # shape 10行1列
        # 初始化抗体浓度
        density = self.calc_density(pop, self.delta)
        # print("初始化抗体浓度：", density)   # shape 10行1列
        # 计算初始抗体种群的激励度：抗体激励度是对抗体质量的最终评价结果，需要综合考虑抗体亲和度和抗体浓度
        simulation = self.calc_simulation(init_simulation, density)
        # print("计算初始抗体种群的激励度：", simulation)
        index = np.argsort(-simulation)
        # np.argsort()将矩阵a按照axis排序，并返回排序后的下标
        # print("对初始抗体种群的激励度进行排序后的下标：", index)
        pop = pop[:, index]
        new_pop = pop.copy()  # 浅拷贝
        # print("排序后的新的抗体种群：", new_pop)
        simulation = simulation[index]
        # print("排序后的激励度,与排序后的新的抗体种群对应:", simulation)
        # 免疫循环
        for gen in range(self.G):

            best_a = np.zeros([self.dim, int(self.pop_size / 2)])  # 用于保留每次克隆后亲和度最高的个体
            best_a_simulation = np.zeros([int(self.pop_size / 2), ])  # 保存激励度

            # 免疫选择算子：根据抗体的激励度确定选择哪些抗体进入克隆选择操作，这一步就是进行个体选择的
            # 在抗体种群中激励度高的抗体具有更好的质量，更有可能被选中进行克隆操作，在搜索空间中更有搜索价值
            # 选出激励度前50%的个体进行免疫
            for i in range(int(self.pop_size / 2)):
                a = new_pop[:, i].reshape([2, 1])
                # print("免疫选择的结果：", a)  # 此时取得是new_pop 的前50%个个体，在循环内部，a是一个个体，即5行两列
                # 克隆算子：将免疫选择算子选中的抗体个体进行复制
                b = np.tile(a, (1, self.colone_num))  # 克隆10份
                # print("克隆结果b", b)  #
                # 变异算子：变异算子对克隆算子得到的抗体克隆结果进行变异操作
                bianyi_jieguo = []
                for j in range(self.colone_num):  # 将会产生10个变异体
                    self.mutate(a, gen, self.mutate_rate, self.dim, self.limit_x, self.limit_y)
                    bianyi_jieguo.append(a)
                # print("变异结果：", bianyi_jieguo)
                b[:, 0] = pop[:, i]  # 保留克隆源个体,里面是原个体
                # print("保留克隆源个体,里面是原个体", b[:, 0])
                # 保留亲和度最高的个体
                b_simulation = self.func(b[0, :], b[1, :])
                index = np.argsort(-b_simulation)
                # print("保留亲和度最高的个体:", b_simulation)
                # print("index", index)  [0,1,2,3,4,5,6,7,8,9]
                # 最佳个体亲和度
                best_a_simulation = b_simulation[index][0]
                # 最佳个体
                best_a[:, i] = b[:, index][:, 0]  # 5行2列
            # print("最佳个体亲和度：", best_a_simulation)
            # print("最佳个体：", best_a[:])

            # 计算免疫种群的激励度
            a_density = self.calc_density(best_a, self.delta)
            best_a_simulation = self.calc_simulation(best_a_simulation, a_density)

            # 种群刷新:种群刷新算子用于对种群中激励度较低的抗体进行刷新，从抗体种群中产出激励度低的个体，并以随机生成的新抗体代替
            new_a = self.init_pop(2, int(self.pop_size / 2), self.limit_x, self.limit_y)

            # 新生种群激励度
            new_a_simulation = self.func(new_a[0, :], new_a[1, :])
            new_a_density = self.calc_density(new_a, self.delta)
            new_a_simulation = self.calc_simulation(new_a_simulation, new_a_density)

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
        for i in range(int(self.pop_size / 2)):
            ax.scatter(pop[0, i], pop[1, i], self.func(pop[0, i], pop[1, i]), color='red')
            print('最优解:', 'x = ', pop[0, i], 'y = ', pop[1, i], end='\n')
            print('结果:', 'z = ', self.func(pop[0, i], pop[1, i]))
        x = np.arange(self.limit_x[0], self.limit_x[1], (self.limit_x[1] - self.limit_x[0]) / 50)
        y = np.arange(self.limit_y[0], self.limit_y[1], (self.limit_y[1] - self.limit_y[0]) / 50)
        x, y = np.meshgrid(x, y)
        z = self.func(x, y)
        max_z = np.max(z)
        print("max_z:", max_z)
        ax.plot_surface(x, y, z, rstride=1, cstride=1, color='green', alpha=0.5)
        plt.show()


if __name__ == "__main__":
    ia = IA()
    ia.run()
