import numpy as np
import numpy.random as random
from matplotlib.pyplot import plot, xlim, ylim, xlabel, ylabel, savefig
import os
import imageio.v2 as imageio
from measures.measures import Dirac, Measure
from copy import deepcopy as copy


class BirthLocalMutation:

    def __init__(self, beta, diracNumber=1):
        self.init = [[0, 1]]
        self.beta = beta
        self.diracNumber = diracNumber
        self.process = [self.init.copy()]

    def simulation(self, tmax: int):
        self.t = [0]
        ttot = 0
        for n in range(tmax):
            self.event(self.birthRates())
        """while ttot <= tmax:
            nbrInd = self.individualNumber()
            if nbrInd <= 0:
                self.t.append(tmax)
                self.process.append([[0, 0]])
                break
            else:
                birthRates = self.birthRates()
                naissance = sum(birthRates)
                tpsArret = random.exponential(1 / naissance)
                ttot = ttot + tpsArret
                if ttot <= tmax:
                    self.t.append(ttot)
                    self.event(birthRates)"""

    def individualNumber(self):
        actualPopulation = self.process[-1]
        return sum([population[1] for population in actualPopulation])

    def birthRates(self):
        actualPopulation = self.process[-1]
        size = len(actualPopulation)
        rates = np.zeros(size)
        for k in range(size):
            rates[k] = actualPopulation[k][1] * np.exp(-actualPopulation[k][0])
        return rates

    def event(self, birthRates):
        actualPopulation = copy(self.process[-1])
        totRate = sum(birthRates)
        sumRate = 0
        unif = random.random()
        for k in range(len(birthRates)):
            sumRate += birthRates[k] / totRate
            if unif <= sumRate:
                parent = copy(actualPopulation[k])
                index = k
                break
        unif = random.random()
        newList = copy(actualPopulation)
        if unif <= self.beta:
            if index + 1 == len(birthRates):
                newList.append([parent[0] + self.diracNumber, 1])
            else:
                print(k, len(newList))
                newList[k + 1][1] += 1
        else:
            newList[k][1] += 1
        self.process.append(newList)

    def anim(self, path="anim.gif"):
        if not os.path.exists("GIF"):
            os.makedirs("GIF")
        maxDirac = self.process[-1][-1][0]
        filenames = []
        for k in range(len(self.process)):
            µ = sum([d[1] * Dirac(d[0], [0, maxDirac]) for d in self.process[k]])
            µ = µ.normalize()
            ylim(0, 1)
            µ.save(f"GIF//{k}.png", color="green")
            print(k)
            filenames.append(f"GIF//{k}.png")
        with imageio.get_writer(path, mode="I", duration=5, loop=False) as writer:
            for filename in filenames:
                # read the picture
                image = imageio.imread(filename)
                # add the picture to the gif
                writer.append_data(image)
                # delete the picture
                os.remove(filename)

    def plot(self, path="tkt.png"):
        maxDirac = self.process[-1][-1][0]
        µ = sum([d[1] * Dirac(d[0], [0, maxDirac]) for d in self.process[-1]])
        µ.save(path, color="green")
