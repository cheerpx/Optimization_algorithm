import numpy as np
import copy
import matplotlib.pyplot as plt
from B07_AFSIndividual import AFSIndividual
"""
人工鱼群算法步骤：
（1）初始化：
    初始化参数：种群规模popsize;人工鱼的视野Visual;人工鱼的步长step;拥挤度因子;重复次数
    初始化鱼群：
（2）计算鱼群中每条鱼的适应度，并将鱼群中最优适应度值以及对应的鱼的位置保存，我们称之为 公告牌
（3）鱼群进行 聚群行为或者跟随行为（选择那种行为根据聚群行为和跟随行为发生之后的适应度值来决定）：
        如果聚群行为之后的适应度值大，该鱼就进行聚群行为
        如果跟随行为之后的适应度值大，该鱼就进行跟随行为
    （3.1）聚群行为：人工鱼探索当前邻域内的伙伴数量，并计算伙伴的中心位置，然后把新得到的中心位置的目标函数与当前位置的目标函数相比较，
            如果中心位置的目标函数优于当前位置的目标函数并且不是很拥挤，则当前位置向中心位置移动一步，否则执行觅食行为。
    （3.1）跟随行为：人工鱼搜索周围邻域内鱼的最优位置，当最优位置的目标函数值大于当前位置的目标函数值并且不是很拥挤，则当前位置向最优邻域鱼移动一步，
                否则执行觅食行为。
（4）选择某种行为之后，计算种群适应度，并将鱼群中最优适应度值与公告牌比较，如果优于公告牌，则更新公告牌的状态
（5）判断是否满足终止条件，如果满足，输出最优解，如果不满足，继续（3）（4）

"""

class ArtificialFishSwarm:
    def __init__(self, sizepop, vardim, bound, MAXGEN, params):
        # params[0]=visual  params[1]=step  params[2]=拥挤度因子  params[3]=重复次数
        # fa = ArtificialFishSwarm(60, 25, bound, 1000, [0.001, 0.0001, 0.618, 40])
        self.sizepop = sizepop
        self.vardim = vardim
        self.bound = bound
        self.MAXGEN = MAXGEN
        self.params = params
        self.population = []
        self.fitness = np.zeros((self.sizepop, 1))
        self.trace = np.zeros((self.MAXGEN, 2))
        self.lennorm = 6000

    # 初始化种群
    def initialize(self):
        for i in range(0, self.sizepop):
            ind = AFSIndividual(self.vardim, self.bound)
            ind.generate()
            self.population.append(ind)


    def evaluation(self, x):
        x.calculateFitness()

    # 人工鱼觅食行为
    def forage(self, x):
        newInd = copy.deepcopy(x)
        found = False
        # 觅食行为：设置人工鱼当前的状态，并在其感知范围内随机选择另一个状态，
        for i in range(0, self.params[3]):
            indi = self.randSearch(x, self.params[0])  # 并在其感知范围内随机选择另一个状态，
            # 如果得到的新状态indi的目标函数大于当前的状态，则向新选择得到的状态靠近一步，反之，重新随机选择状态，如此循环
            if indi.fitness > x.fitness:
                newInd.individual = x.individual + np.random.random(self.vardim) * self.params[1] * self.lennorm * (
                        indi.individual - x.individual) / np.linalg.norm(indi.individual - x.individual)
                newInd = indi
                found = True
                break
        # 如果选择次数达到一定数量之后，状态仍没有改变，则随机移动一步
        if not (found):
            newInd = self.randSearch(x, self.params[1])
        return newInd

    # 人工鱼随机行为
    def randSearch(self, x, searLen):

        ind = copy.deepcopy(x)
        ind.individual += np.random.uniform(-1, 1, self.vardim) * searLen * self.lennorm
        for j in range(0, self.vardim):
            if ind.individual[j] < self.bound[0, j]:
                ind.individual[j] = self.bound[0, j]
            if ind.individual[j] > self.bound[1, j]:
                ind.individual[j] = self.bound[1, j]
        self.evaluation(ind)
        return ind

    # 人工鱼聚群行为
    def huddle(self, x):
        newInd = copy.deepcopy(x)
        dist = self.distance(x)
        index = []
        for i in range(1, self.sizepop):
            if dist[i] > 0 and dist[i] < self.params[0] * self.lennorm:
                index.append(i)
        nf = len(index)
        if nf > 0:
            xc = np.zeros(self.vardim)
            for i in range(0, nf):
                xc += self.population[index[i]].individual
            xc = xc / nf
            cind = AFSIndividual(self.vardim, self.bound)
            cind.individual = xc
            cind.calculateFitness()
            # 聚集行为： 人工鱼探索当前邻域内的伙伴数量，并计算伙伴的中心位置，然后把新得到的中心位置的目标函数与当前位置的目标函数相比较

            # 如果中心位置的目标函数 优于当前位置的目标函数  并且不是很拥挤，那么当前位置向中心位置移动一步
            if (cind.fitness / nf) > (self.params[2] * x.fitness):
                xnext = x.individual + np.random.random(
                    self.vardim) * self.params[1] * self.lennorm * (xc - x.individual) / np.linalg.norm(xc - x.individual)
                for j in range(0, self.vardim):
                    if xnext[j] < self.bound[0, j]:
                        xnext[j] = self.bound[0, j]
                    if xnext[j] > self.bound[1, j]:
                        xnext[j] = self.bound[1, j]
                newInd.individual = xnext
                self.evaluation(newInd)
                print("hudding")
                return newInd
            # 否则：即中心位置的目标函数 不优于当前位置的目标函数，那么当前位置不移动，而是执行觅食行为
            else:
                return self.forage(x)
        else:
            return self.forage(x)

    # 人工鱼追尾行为
    def follow(self, x):
        newInd = copy.deepcopy(x)
        dist = self.distance(x)
        index = []
        for i in range(1, self.sizepop):
            if dist[i] > 0 and dist[i] < self.params[0] * self.lennorm:
                index.append(i)
        nf = len(index)
        if nf > 0:
            best = -999999999.
            bestIndex = 0
            for i in range(0, nf):
                if self.population[index[i]].fitness > best:
                    best = self.population[index[i]].fitness
                    bestIndex = index[i]
            # 人工鱼搜索周围邻域内鱼的最优位置，当最优位置的目标函数值大于当前位置的目标函数值并且不是很拥挤，
            # 那么当前位置向最优邻域鱼移动一步
            if (self.population[bestIndex].fitness / nf) > (self.params[2] * x.fitness):
                xnext = x.individual + np.random.random(
                    self.vardim) * self.params[1] * self.lennorm * (
                                self.population[bestIndex].individual - x.individual) / np.linalg.norm(
                    self.population[bestIndex].individual - x.individual)
                for j in range(0, self.vardim):
                    if xnext[j] < self.bound[0, j]:
                        xnext[j] = self.bound[0, j]
                    if xnext[j] > self.bound[1, j]:
                        xnext[j] = self.bound[1, j]
                newInd.individual = xnext
                self.evaluation(newInd)
                # print "follow"
                return newInd
            # 否则，即人工鱼搜索周围邻域内鱼的最优位置，当最优位置的目标函数值不大于当前位置的目标函数值，执行觅食行为
            else:
                return self.forage(x)
        else:
            return self.forage(x)

    def solve(self):
        self.t = 0
        self.initialize()
        print("产生的初始化种群：", self.population)  # 60 sizepop 行 25 vardim 列
        # 评估人工鱼群
        for i in range(0, self.sizepop):
            self.evaluation(self.population[i])
            # 计算每条鱼的适应度值
            self.fitness[i] = self.population[i].fitness
        bestIndex = np.argmax(self.fitness)  # 用于返回一个numpy数组中最大值的索引值，相当于公告牌
        self.best = copy.deepcopy(self.population[bestIndex])   # 相当于公告牌
        self.avefitness = np.mean(self.fitness)
        self.trace[self.t, 0] = (1 - self.best.fitness) / self.best.fitness
        self.trace[self.t, 1] = (1 - self.avefitness) / self.avefitness
        print("Generation %d: optimal function value is: %f; average function value is %f" % (self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
        while self.t < self.MAXGEN - 1:
            # 迭代循环开始
            self.t += 1
            # newpop = []
            for i in range(0, self.sizepop):
                xi1 = self.huddle(self.population[i])  # 聚群行为，对于每一条鱼让其进行聚群
                xi2 = self.follow(self.population[i])   # 跟随行为，对于每一条鱼让其进行跟随
                # 将聚群行为和跟随行为产生的适应度进行对比
                if xi1.fitness > xi2.fitness:    # 如果聚群行为得到的适应度值更大，当前鱼选择聚群行为
                    self.population[i] = xi1
                    self.fitness[i] = xi1.fitness
                else:                              # 否则当前鱼i选择跟随行为
                    self.population[i] = xi2
                    self.fitness[i] = xi2.fitness
            best = np.max(self.fitness)
            bestIndex = np.argmax(self.fitness)   # 记录当前鱼群中最优适应度值
            if best > self.best.fitness:   # 如果当前代的鱼群的最优适应度比公告牌中的还优
                self.best = copy.deepcopy(self.population[bestIndex])   # 那么就将公告牌中的状态和值更换
            self.avefitness = np.mean(self.fitness)
            self.trace[self.t, 0] = (1 - self.best.fitness) / self.best.fitness
            self.trace[self.t, 1] = (1 - self.avefitness) / self.avefitness
            print("Generation %d: optimal function value is: %f; average function value is %f" % (
                self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
        print("Optimal function value is: %f; " % self.trace[self.t, 0])
        print("Optimal solution is:")
        print(self.best.individual)
        self.printResult()

    def distance(self, x):
        dist = np.zeros(self.sizepop)
        for i in range(0, self.sizepop):
            dist[i] = np.linalg.norm(x.individual - self.population[i].individual) / 6000
        return dist

    def printResult(self):
        x = np.arange(0, self.MAXGEN)
        y1 = self.trace[:, 0]
        y2 = self.trace[:, 1]
        plt.plot(x, y1, 'r', label='optimal value')
        plt.plot(x, y2, 'g', label='average value')
        plt.xlabel("Iteration")
        plt.ylabel("function value")
        plt.title("Artificial Fish Swarm algorithm for function optimization")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    bound = np.tile([[-600], [600]], 25)
    fa = ArtificialFishSwarm(60, 25, bound, 1000, [0.001, 0.0001, 0.618, 40])
    fa.solve()
