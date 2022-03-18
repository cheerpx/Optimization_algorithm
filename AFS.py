import numpy as np
import ObjFunction
import copy


class AFSIndividual:

    """class for AFSIndividual"""

    def __init__(self, vardim, bound):
        '''
        vardim: dimension of variables
        bound: boundaries of variables
        '''
        self.vardim = vardim
        self.bound = bound

    def generate(self):
        '''
        generate a rondom chromsome
        '''
        len = self.vardim
        rnd = np.random.random(size=len)
        self.chrom = np.zeros(len)
        self.velocity = np.random.random(size=len)
        for i in range(0, len):
            self.chrom[i] = self.bound[0, i] + \
                (self.bound[1, i] - self.bound[0, i]) * rnd[i]
        self.bestPosition = np.zeros(len)
        self.bestFitness = 0.

    def calculateFitness(self):
        '''
        calculate the fitness of the chromsome
        '''
        self.fitness = ObjFunction.GrieFunc(
            self.vardim, self.chrom, self.bound)


import numpy as np
import random
import copy
import matplotlib.pyplot as plt


class ArtificialFishSwarm:

    """class for  ArtificialFishSwarm"""

    def __init__(self, sizepop, vardim, bound, MAXGEN, params):
        '''
        sizepop: population sizepop
        vardim: dimension of variables
        bound: boundaries of variables, 2*vardim
        MAXGEN: termination condition
        params: algorithm required parameters, it is a list which is consisting of[visual, step, delta, trynum]
        '''
        self.sizepop = sizepop
        self.vardim = vardim
        self.bound = bound
        self.MAXGEN = MAXGEN
        self.params = params
        self.population = []
        self.fitness = np.zeros((self.sizepop, 1))
        self.trace = np.zeros((self.MAXGEN, 2))
        self.lennorm = 6000

    def initialize(self):
        '''
        initialize the population of afs
        '''
        for i in range(0, self.sizepop):
            ind = AFSIndividual(self.vardim, self.bound)
            ind.generate()
            self.population.append(ind)

    def evaluation(self, x):
        '''
        evaluation the fitness of the individual
        '''
        x.calculateFitness()

    def forage(self, x):
        '''
        artificial fish foraging behavior
        '''
        newInd = copy.deepcopy(x)
        found = False
        for i in range(0, self.params[3]):
            indi = self.randSearch(x, self.params[0])
            if indi.fitness > x.fitness:
                newInd.chrom = x.chrom + np.random.random(self.vardim) * self.params[1] * self.lennorm * (
                    indi.chrom - x.chrom) / np.linalg.norm(indi.chrom - x.chrom)
                newInd = indi
                found = True
                break
        if not (found):
            newInd = self.randSearch(x, self.params[1])
        return newInd

    def randSearch(self, x, searLen):
        '''
        artificial fish random search behavior
        '''
        ind = copy.deepcopy(x)
        ind.chrom += np.random.uniform(-1, 1,
                                       self.vardim) * searLen * self.lennorm
        for j in range(0, self.vardim):
            if ind.chrom[j] < self.bound[0, j]:
                ind.chrom[j] = self.bound[0, j]
            if ind.chrom[j] > self.bound[1, j]:
                ind.chrom[j] = self.bound[1, j]
        self.evaluation(ind)
        return ind

    def huddle(self, x):
        '''
        artificial fish huddling behavior
        '''
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
                xc += self.population[index[i]].chrom
            xc = xc / nf
            cind = AFSIndividual(self.vardim, self.bound)
            cind.chrom = xc
            cind.calculateFitness()
            if (cind.fitness / nf) > (self.params[2] * x.fitness):
                xnext = x.chrom + np.random.random(
                    self.vardim) * self.params[1] * self.lennorm * (xc - x.chrom) / np.linalg.norm(xc - x.chrom)
                for j in range(0, self.vardim):
                    if xnext[j] < self.bound[0, j]:
                        xnext[j] = self.bound[0, j]
                    if xnext[j] > self.bound[1, j]:
                        xnext[j] = self.bound[1, j]
                newInd.chrom = xnext
                self.evaluation(newInd)
                # print "hudding"
                return newInd
            else:
                return self.forage(x)
        else:
            return self.forage(x)

    def follow(self, x):
        '''
        artificial fish following behivior
        '''
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
            if (self.population[bestIndex].fitness / nf) > (self.params[2] * x.fitness):
                xnext = x.chrom + np.random.random(
                    self.vardim) * self.params[1] * self.lennorm * (self.population[bestIndex].chrom - x.chrom) / np.linalg.norm(self.population[bestIndex].chrom - x.chrom)
                for j in range(0, self.vardim):
                    if xnext[j] < self.bound[0, j]:
                        xnext[j] = self.bound[0, j]
                    if xnext[j] > self.bound[1, j]:
                        xnext[j] = self.bound[1, j]
                newInd.chrom = xnext
                self.evaluation(newInd)
                # print "follow"
                return newInd
            else:
                return self.forage(x)
        else:
            return self.forage(x)

    def solve(self):
        '''
        evolution process for afs algorithm
        '''
        self.t = 0
        self.initialize()
        # evaluation the population
        for i in range(0, self.sizepop):
            self.evaluation(self.population[i])
            self.fitness[i] = self.population[i].fitness
        best = np.max(self.fitness)
        bestIndex = np.argmax(self.fitness)
        self.best = copy.deepcopy(self.population[bestIndex])
        self.avefitness = np.mean(self.fitness)
        self.trace[self.t, 0] = (1 - self.best.fitness) / self.best.fitness
        self.trace[self.t, 1] = (1 - self.avefitness) / self.avefitness
        print("Generation %d: optimal function value is: %f; average function value is %f" % (
            self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
        while self.t < self.MAXGEN - 1:
            self.t += 1
            # newpop = []
            for i in range(0, self.sizepop):
                xi1 = self.huddle(self.population[i])
                xi2 = self.follow(self.population[i])
                if xi1.fitness > xi2.fitness:
                    self.population[i] = xi1
                    self.fitness[i] = xi1.fitness
                else:
                    self.population[i] = xi2
                    self.fitness[i] = xi2.fitness
            best = np.max(self.fitness)
            bestIndex = np.argmax(self.fitness)
            if best > self.best.fitness:
                self.best = copy.deepcopy(self.population[bestIndex])
            self.avefitness = np.mean(self.fitness)
            self.trace[self.t, 0] = (1 - self.best.fitness) / self.best.fitness
            self.trace[self.t, 1] = (1 - self.avefitness) / self.avefitness
            print("Generation %d: optimal function value is: %f; average function value is %f" % (
                self.t, self.trace[self.t, 0], self.trace[self.t, 1]))

        print("Optimal function value is: %f; " % self.trace[self.t, 0])
        print("Optimal solution is:")
        print(self.best.chrom)
        self.printResult()

    def distance(self, x):
        '''
        return the distance array to a individual
        '''
        dist = np.zeros(self.sizepop)
        for i in range(0, self.sizepop):
            dist[i] = np.linalg.norm(x.chrom - self.population[i].chrom) / 6000
        return dist

    def printResult(self):
        '''
        plot the result of afs algorithm
        '''
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