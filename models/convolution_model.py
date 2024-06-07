import time as tea

import numpy as np
from matplotlib.pyplot import plot, scatter, show, xlim, ylim

from measures.measures import Dirac, Id, Measure
from models.selection_mutation_model import SelectionMutation


def equation(pn, beta, j, I):
    µ = beta * (pn & j) + (1 - beta) * Measure(
        lambda x: np.exp(-x), I
    ) * pn / pn.integrate(lambda y: np.exp(-y))
    return µ


class Cmodel(SelectionMutation):
    def __init__(self, p0, beta, j, I=[0, 10]):
        super().__init__(equation, p0, [beta, j, I])
        if I[0] != 0:
            print("Warning : Lower bound not equals to zero.")

    def getBeta(self):
        self.getParam(0)

    def getJ(self):
        self.getParam(1)

    def getI(self):
        self.getParam(2)

    def getLimit(self):
        beta, j, I = self.getParams()
        if not isinstance(j, Dirac):
            print("Error : j is not a Dirac.")
            return
        elif j.number > I[1]:
            print("Error : Interval too small.")
            return
        else:
            eta0 = self.getInit().getMinSupport()
            a = j.number
            prod = 1
            N = 300
            sump0 = 0
            for i in range(1, N):
                prod *= 1 - np.exp(-i * a)
                sump0 += beta ** (i) / prod
            p0 = 1 / (1 + sump0)

            size = int((I[1] - eta0) / a + 1)
            ps = np.zeros(size)
            ps[0] = p0
            for i in range(1, size):
                ps[i] = ps[i - 1] * beta / (1 - np.exp(-i * a))
            limit = sum(ps[k] * Dirac(a * k + eta0, I) for k in range(size))

            return limit


def main():
    start = tea.time()

    I = [0, 20]

    def f(x):
        return (x >= 1) * np.exp(x)

    p0 = Measure(f, I).normalize()
    j = Dirac(2.5)
    beta = 0.5
    model = Cmodel(p0, beta, j, I)
    model.animation(path="figures/cmodel.gif", N=50)
    # model.getLimit()
    print("temps d'execution :", tea.time() - start)
    return


""" def equationNul(pn, beta, j):
    return (beta*(pn & j) +
            (1-beta)*(-IdNul(I)).exp() * pn/pn.integrate(lambda y: np.exp(-y)))


p0 = Measure(lambda x: (0 <= x)*(x <= 10), I)
p0 = p0/p0.integrate()
# p0 = Dirac(0)

j = Dirac(1, I)
j = j/j.integrate()

beta = 0.5 """

"""(beta*(pn & j) +
            (1-beta)*(-Id(I)).exp() * pn/pn.integrate(lambda y: np.exp(-y)))"""


# cccccc1 = SelectionMutation(equation, p0, [beta, j])

# cccccc1.timeDifferenceBetweenTwoMeasures(Measure,
#                                          MeasureNul,
#                                          lambda x: (0 <= x)*(x <= 10),
#                                          [0.5, (1,)],
#                                          [0, 100],
#                                          equation,
#                                          equationNul,
#                                          20,
#                                          "Computation with memoization",
#                                          "Computation without memoization",
#                                          Dirac,
#                                          DiracNul)

""" 
model = SelectionMutation(equation, p0, [beta, j]) """

# p = model.convergence(30)
# print("convergence done")

# print(p.integrate())

""" N = 22
pis = np.array([p(i) for i in range(N)])
p = p/sum(pis) """


""" print("p0 : ",p0)
scatter(0, p0)
xlim(-0.1, 10)
p.save() """

""" # creation of the function into the algorithm
def iterativeSave(measure, i):
    # saving the measure's plot
    scatter([k for k in range(11)], ps, color="red")
    xlim(-0.1, I[1])
    µ = measure/sum([measure(k) for k in range(int(I[1])+1)])
    µ.save(filename=f'GIF//{i}.png',
                    label=f'iteration {i}',
                    color="green",
                    loc="upper left",
                    setxlim=[-0.1, I[1]])

# call the algorithm
N = 50
measure = model.convergenceWithFunction(iterativeSave, N=N, k=1)
# save one last time
iterativeSave(measure, N)

# get every file name that have been saved
filenames = [f'GIF//{i}.png' for i in range(N+1)]
# creating the gif
model.gifCreation("resultats/convergence_cmodel.gif", filenames, False)


# Cp = 0
# for i in range(N):
#     Cp += pis[i]*np.exp(-i)
# print(Cp) """
