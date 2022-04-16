import ObjFunction
import numpy as np


class AFSIndividual:

    def __init__(self, vardim, bound):
        self.vardim = vardim
        self.bound = bound

    def generate(self):
        rnd = np.random.random(size=self.vardim)
        self.individual = np.zeros(self.vardim)
        # 随机产生一个个体
        for i in range(0, self.vardim):
            self.individual[i] = self.bound[0, i] + (self.bound[1, i] - self.bound[0, i]) * rnd[i]
        #print("individual:", self.individual)
        self.bestPosition = np.zeros(self.vardim)
        self.bestFitness = 0.

    def calculateFitness(self):
        self.fitness = ObjFunction.GrieFunc(self.vardim, self.individual, self.bound)
