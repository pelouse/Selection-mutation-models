import time as tea

import numpy as np
from matplotlib.pyplot import plot, scatter, show, xlim, ylim

from measures.measures import Dirac, Id, Measure, SumDirac
from models.selection_mutation_model import SelectionMutation


def equation(pn, beta, j, I):
    µ = beta * (pn & j) + (1 - beta) * Measure(
        lambda x: np.exp(-x), I
    ) * pn / pn.integrate(lambda y: np.exp(-y))
    return µ


class Cmodel(SelectionMutation):
    def __init__(self, p0, beta, j, I=[0, 10]):
        super().__init__(equation, p0, [beta, j, I])

    def getBeta(self):
        return self.getParam(0)

    def getJ(self):
        return self.getParam(1)

    def getI(self):
        return self.getParam(2)

    def getLimit(self):
        beta, j, I = self.getParams()
        if not isinstance(j, Dirac):
            print("Error : j is not a Dirac.")
            return
        elif j.number > I[1]:
            print("Error : Interval too small.")
            return
        else:
            eta0 = self.getInit().getMinSupport(1e-2)
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

    def diracDistinction(self, dirac: SumDirac):
        a = self.getJ().number
        independantMeasures = []
        independantNumbers = []
        numbers = dirac.getDiracNumbers()
        saved = False
        for d in dirac.List:
            for k in range(len(independantMeasures)):
                if d.number == independantNumbers[k][-1] + a:
                    independantMeasures[k].append(d)
                    independantNumbers[k].append(d.number)
                    saved = True
            if not saved:
                independantMeasures.append([d])
                independantNumbers.append([d.number])
            saved = False
        return independantMeasures

    def iteration(self, measure: Measure):
        if not measure.isDirac():
            return super().iteration(measure)
        else:
            beta, j, I = self.getParams()
            eta0 = self.getInit().getMinSupport(1e-2)
            a = j.number
            if isinstance(measure, Dirac):
                numberMaxOfDirac = int((I[1] - eta0) / a + 1)
                pn = np.zeros(numberMaxOfDirac)
                pn[0] = 1
                wn = sum(
                    np.exp(-(a * k + eta0)) * pn[k] for k in range(numberMaxOfDirac)
                )
                pn1 = np.zeros(numberMaxOfDirac)
                pn1[0] = (1 - beta) * np.exp(-(eta0)) * pn[0] / wn
                for k in range(numberMaxOfDirac - 1):
                    pn1[k + 1] = (
                        beta * pn[k]
                        + (1 - beta) * np.exp(-(a * k + a + eta0)) * pn[k + 1] / wn
                    )

                return sum(
                    pn1[k] * Dirac(a * k + eta0, I) for k in range(numberMaxOfDirac)
                )
            else:
                wn = sum(
                    [
                        np.exp(-dirac.number) * dirac(dirac.number)
                        for dirac in measure.List
                    ]
                )
                distinction = self.diracDistinction(measure)
                returnMeasure = 0
                for iteration in distinction:
                    pn = [dirac(dirac.number) for dirac in iteration]
                    pn1 = np.zeros(len(pn) + (iteration[-1].number + a <= I[1]))
                    pn1[0] = (1 - beta) * np.exp(-iteration[0].number) * pn[0] / wn
                    for k in range(len(pn) - 1):
                        pn1[k + 1] = (
                            beta * pn[k]
                            + (1 - beta)
                            * np.exp(-iteration[k + 1].number)
                            * pn[k + 1]
                            / wn
                        )
                    if iteration[-1].number + a <= I[1]:
                        pn1[-1] = beta * pn[-1]
                    returnMeasure += sum(
                        pn1[k] * Dirac(iteration[0].number + a * k, I)
                        for k in range(len(pn1))
                    )
                return returnMeasure
            """ beta, j, I = self.getParams()
            eta0 = self.getInit().getMinSupport(1e-2)
            a = j.number
            numberMaxOfDirac = int((I[1] - eta0) / a + 1)
            parameterDiracList = [eta0 + k * a for k in range(numberMaxOfDirac)]
            pn = np.zeros(numberMaxOfDirac)
            if isinstance(measure, Dirac):
                pn[0] = 1
            else:
                for k in range(numberMaxOfDirac):
                    pn[k] = measure.List[k].function(a * k + eta0)
            wn = sum(np.exp(-(a * k + eta0)) * pn[k] for k in range(numberMaxOfDirac))
            pn1 = np.zeros(numberMaxOfDirac)
            pn1[0] = (1 - beta) * np.exp(-(eta0)) * pn[0] / wn
            for k in range(numberMaxOfDirac - 1):
                pn1[k + 1] = (
                    beta * pn[k]
                    + (1 - beta) * np.exp(-(a * k + a + eta0)) * pn[k + 1] / wn
                )

            return sum(pn1[k] * Dirac(a * k + eta0, I) for k in range(numberMaxOfDirac)) """

    """ def animation(
        self,
        path: str = "mygif.gif",
        N: int = 100,
        loop: bool = True,
        interval: list = None,
    ):
        p0 = self.getInit()
        if not (
            isinstance(p0, Dirac) or (isinstance(p0, SumDirac) and p0.isOnlyDirac())
        ):
            super().animation(path, N, loop, interval)
        else:
            beta, I = self.getBeta(), self.getI()
            limit = self.getLimit()
            parameterDiracList = limit.getDiracNumbers()
            size = len(parameterDiracList)
            self.gifPrevention()
            pn = np.zeros(size)
            if isinstance(p0, Dirac):
                pn[0] = 1
            else:
                numberDiracInit = len(p0.List)
                numberDiracInit = min(numberDiracInit, size)
                for k in range(numberDiracInit):
                    pn[k] = p0.List[k].function(parameterDiracList[k])

            epsilon = 10 * limit.getEpsilon()
            for n in range(N):
                limit.plot(label="limit", color="red")
                measure = sum(
                    pn[k] * Dirac(parameterDiracList[k], I) for k in range(size)
                )
                measure.save(
                    filename=f"GIF//{n}.png",
                    label=f"iteration {n}",
                    color="green",
                    loc="upper left",
                    interval=interval,
                    setxlim=[
                        limit.getInterval()[0] - epsilon,
                        limit.getInterval()[1] + epsilon,
                    ],
                )
                wn = sum(np.exp(-parameterDiracList[k]) * pn[k] for k in range(size))
                pn1 = np.zeros(size)
                pn1[0] = (1 - beta) * np.exp(-parameterDiracList[0]) * pn[0] / wn
                for k in range(size - 1):
                    pn1[k + 1] = (
                        beta * pn[k]
                        + (1 - beta)
                        * np.exp(-parameterDiracList[k + 1])
                        * pn[k + 1]
                        / wn
                    )
                pn = pn1
                print(str(n) + "/" + str(N))
            limit.plot(label="limit", color="red")
            measure = sum(pn[k] * Dirac(parameterDiracList[k]) for k in range(size))
            measure.save(
                filename=f"GIF//{N}.png",
                label=f"iteration {N}",
                color="green",
                loc="upper left",
                interval=interval,
                setxlim=[
                    limit.getInterval()[0] - epsilon,
                    limit.getInterval()[1] + epsilon,
                ],
            )
            print(str(N) + "/" + str(N))
            self.gifCreation(path, [f"GIF/{n}.png" for n in range(N + 1)], loop) """


def animation_convergence(p0, beta, j, I, N=200):
    model = Cmodel(p0, beta, j, I)
    model.animation(path="figures/cmodel/convergence.gif", N=N)
    return


def animation_repartition(p0, beta, j, I):
    model = Cmodel(p0, beta, j, I)
    model.animationRepartition(path="figures/cmodel/repartition.gif", N=50)
    return
