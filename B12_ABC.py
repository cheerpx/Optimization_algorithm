import random, math, copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
"""
def GrieFunc(data):  # 目标函数
    s1 = 0.
    s2 = 1.
    for k, x in enumerate(data):  #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        #产生25个x
        s1 = s1 + x ** 2
        print("k:", k)
        print("x:", x)
        s2 = s2 * math.cos(x / math.sqrt(k + 1))
    y = (1. / 4000.) * s1 - s2 + 1
    return 1. / (1. + y)
"""


def GrieFunc(data):  # 目标函数
    s1 = 0
    for k, x in enumerate(data):  # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        s1 = s1 + x
    y = s1
    return y


class ABSIndividual:  # 初始化，怎么初始化呢？
    def __init__(self, bound):
        self.score = 0.
        self.invalidCount = 0  # 无效次数（成绩没有更新的累积次数）
        # 由于这里的目标函数是一元的，因此这个chrom是一维数组。如果是二元函数，chrom就要变化
        self.chrom = [random.uniform(a, b) for a, b in
                      zip(bound[0, :], bound[1, :])]  # 随机初始化  zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
        # random.uniform(a, b)返回a,b之间的随机浮点数，在本例中，返回-500到500之间的数，若a<=b则范围[a,b]，若a>=b则范围[b,a] ，a和b可以是实数。
        # 即一个chrom是一个有25个数值的数组，这样就完成了初始化，在25维的空间中各产生了一个x，因为目标函数是一元函数。
        # a=[1,2,3]  b=[4,5,6]   zip(a,b) = [(1,4),(2，5),(3,6) ]
        self.calculateFitness()
        # for a, b in zip(bound[0, :], bound[1, :]):  #循环了30遍，正好是初始化蜂群蜜源的大小。
        #    print("a:", a)  #产生25个a    #正好是vardim  ，变量的维度，

    def calculateFitness(self):
        # print("chrom:", len(self.chrom))
        self.score = GrieFunc(self.chrom)  # 计算当前成绩


class ArtificialBeeSwarm:
    def __init__(self, foodCount, onlookerCount, bound, maxIterCount=1000, maxInvalidCount=200):
        # abs = ArtificialBeeSwarm(30, 30, bound, 1000, 200)
        self.foodCount = foodCount  # 蜜源个数，等同于雇佣蜂数目
        self.onlookerCount = onlookerCount  # 观察蜂个数
        self.bound = bound  # 各参数上下界
        self.maxIterCount = maxIterCount  # 迭代次数
        self.maxInvalidCount = maxInvalidCount  # 最大无效次数  以此来判断是否丢失蜜源
        self.foodList = [ABSIndividual(self.bound) for k in
                         range(self.foodCount)]  # 初始化各蜜源，就本例来说，如果设定蜜源个数为30，维度为15，那么这个蜜源应该是在每一维里面都有30个采蜜蜂。
        self.foodScore = [d.score for d in
                          self.foodList]  # 各蜜源最佳成绩  #将每一个蜜源的适应度值按照目标函数计算，保存在foodScore里面，所以foodScore的维度也是30
        # self.bestFood = self.foodList[np.argmax(self.foodScore)]  # 全局最佳蜜源，找最大值
        self.bestFood = self.foodList[np.argmin(self.foodScore)]

    def updateFood(self, i):  # 更新第i个蜜源
        k = random.randint(0, self.bound.shape[1] - 1)  # 随机选择调整参数的维度
        # 下面是寻找新蜜源的公式，ppt第9张。
        j = random.choice([d for d in range(self.foodCount) if d != i])  # random.choice从序列中获取一个随机元素；
        vi = copy.deepcopy(self.foodList[i])
        # print("第i个蜜源的适应度值", foodList[i].score)
        vi.chrom[k] += random.uniform(-1.0, 1.0) * (vi.chrom[k] - self.foodList[j].chrom[k])  # 调整参数

        vi.chrom[k] = np.clip(vi.chrom[k], self.bound[0, k], self.bound[1, k])  # 参数不能越界
        vi.calculateFitness()  # 计算新蜜源的适应度值
        """
        if vi.score > self.foodList[i].score:  # 如果新蜜源的适应度值比当前蜜源好
            self.foodList[i] = vi
            if vi.score > self.foodScore[i]:  # 如果新蜜源的适应度值 比历史成绩好（如重新初始化，当前成绩可能低于历史成绩）
                self.foodScore[i] = vi.score
                if vi.score > self.bestFood.score:  # 如果成绩全局最优
                    self.bestFood = vi
            self.foodList[i].invalidCount = 0   #当前蜜源进行了更新
        else:
            self.foodList[i].invalidCount += 1   #当前蜜源没有进行更新
        """
        if vi.score < self.foodList[i].score:  # 如果新蜜源的适应度值比当前蜜源好
            self.foodList[i] = vi
            if vi.score < self.foodScore[i]:  # 如果新蜜源的适应度值 比历史成绩好（如重新初始化，当前成绩可能低于历史成绩）
                self.foodScore[i] = vi.score
                if vi.score < self.bestFood.score:  # 如果成绩全局最优
                    self.bestFood = vi
            self.foodList[i].invalidCount = 0  # 当前蜜源进行了更新
        else:
            self.foodList[i].invalidCount += 1  # 当前蜜源没有进行更新

    def employedBeePhase(self):  # 雇佣蜂，即采蜜蜂的行为：寻找新蜜源
        for i in range(0, self.foodCount):  # 各蜜源依次更新
            self.updateFood(i)  # 从初始化的那群采蜜蜂开始，产生新的解，即产生新的蜜源

    def onlookerBeePhase(self):  # 跟随蜂，即观察蜂的行为：应用轮盘赌算法选取蜜源，
        foodScore = [d.score for d in self.foodList]
        maxScore = np.min(foodScore)
        accuFitness = [(0.9 * d / maxScore + 0.1, k) for k, d in enumerate(foodScore)]  # 得到各蜜源的 相对分数和索引号
        # print("得到各蜜源的相对分数和索引号", accuFitness)
        for k in range(0, self.onlookerCount):
            i = random.choice([d[1] for d in accuFitness if
                               d[0] >= random.random()])  # 随机从相对分数大于随机门限的蜜源中选择跟随,选中蜜源i,同时更新蜜源i，即获得蜜源周围的新蜜源，然后与此比较
            self.updateFood(i)

    def scoutBeePhase(self):  # 探索蜂，即侦查蜂的行为：用来侦查某一蜜源是否应该丢弃，比如蜜源i应该被丢弃，那么，重新对这个蜜源进行初始化，以及记录新的适应度值
        for i in range(0, self.foodCount):
            if self.foodList[i].invalidCount > self.maxInvalidCount:  # 如果该蜜源没有更新的次数超过指定门限，则重新初始化
                self.foodList[i] = ABSIndividual(self.bound)
                self.foodScore[i] = min(self.foodScore[i], self.foodList[i].score)

    def solve(self):
        trace = []
        trace.append((self.bestFood.score, np.mean(self.foodScore)))
        for k in range(self.maxIterCount):  # 迭代次数，每迭代一次：就进行这三种动作，三种行为结束之后，将当前的最优适应度值存进trace中，
            self.employedBeePhase()
            self.onlookerBeePhase()
            self.scoutBeePhase()
            trace.append((self.bestFood.score, np.mean(self.foodScore)))
        print(self.bestFood.score)
        self.printResult(np.array(trace))

    def printResult(self, trace):
        x = np.arange(0, trace.shape[0])
        plt.plot(x, [(1 - d) / d for d in trace[:, 0]], 'r', label='optimal value')
        plt.plot(x, [(1 - d) / d for d in trace[:, 1]], 'g', label='average value')
        plt.xlabel("Iteration")
        plt.ylabel("function value")
        plt.title("Artificial Bee Swarm algorithm for function optimization")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    random.seed()
    vardim = 7  # 和向目标函数中传入的data的维度一样
    bound = np.tile([[-500], [500]], vardim)
    abs = ArtificialBeeSwarm(30, 30, bound, 1000, 500)
    abs.solve()
