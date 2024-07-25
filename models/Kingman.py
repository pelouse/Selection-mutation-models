import numpy as np
from matplotlib.pyplot import (
    close,
    legend,
    plot,
    savefig,
    title,
    xlabel,
    xlim,
    ylim,
    axvline,
)

from measures.measures import Dirac, Id, Measure, OneMeasure
from models.selection_mutation_model import SelectionMutation


# kingman equation
def kingmanEq(p, beta, q):
    return ((1 - beta) / p.getMean()) * Id() * p + beta * q


class Kingman(SelectionMutation):

    def __init__(self, p0, beta, q):
        """
        Constructor

        Parameters
        ----------
        p0 : Measure
            The initial measure of pn.
        beta : float
            The mutation probability between 0 and 1.
        q : Measure
            The mutation distribution.
        """
        super().__init__(kingmanEq, p0, [beta, q])
        # the convergence case depending on the parameters
        self.case = None

    def getBeta(self):
        return self.getParam(0)

    def getQ(self):
        return self.getParam(1)

    def getP0(self):
        return self.getInit()

    def getAll(self):
        return self.getP0(), self.getBeta(), self.getQ()

    def setBeta(self, value):
        self.setParam(value, 0)
        # the case must be recalculated
        self.case = None

    def setQ(self, measure):
        self.setParam(measure, 1)
        # the case must be recalculated
        self.case = None

    def setP0(self, measure):
        self.setInit(measure)
        # the case must be recalculated
        self.case = None

    def getY0(self):
        """
        compute the y0 to get the theorical limit.
        Return y0
        """
        # get the parameters
        p0, beta, q = self.getAll()
        # get the support of p0
        eta = p0.getMaxSupport()
        y0 = eta
        for x0 in np.arange(eta, 1000, 0.001):
            # we take the first value such that the function is less than 1
            if self.poleFun(x0) <= 1:
                y0 = x0
                break
        return y0

    def getInterval(self, cut: float = 0.05):
        """
        Get the convergence interval of the kingman model.
        It depends on if there is a Dirac in the limit expression.

        Parameters
        ----------
        cut : float, optional
            In case of the dirac, the convergence is on [0, eta0[. We want
            a number near eta0 but not equal. We so take eta0 - cut.
            The default is 0.05.

        Returns
        -------
        list
            The interval where we can plot to see the convergence.
        """
        p0 = self.getP0()
        eta0 = p0.getMaxSupport()
        # check the case
        if self.poleFun(eta0) <= 1:
            return [0, eta0 - cut]
        else:
            return [0, 1]

    def getLimit(self):
        """
        Get the limit of pn.

        Returns
        -------
        p : Measure
            The limit measure of pn.
        """
        # qet the parameters
        p0, beta, q = self.getAll()
        # get the case
        if self.poleFun(p0.getMaxSupport()) < 1:
            y0 = self.getY0()
            # compute of pi0
            pi0 = 1 - q.integrate(
                lambda x: beta / (1 - x / y0), interval=[0, q.getMaxSupport()]
            )
            # return the limit

            def fun(x):
                if x < y0:
                    return 1 / (1 - x / y0)
                else:
                    return 0

            f = np.vectorize(fun)
            return beta * q * Measure(f) + pi0 * Dirac(p0.getMaxSupport())
        else:
            # pi0 = 0 so there is no Dirac
            return beta * q / (1 - Id() / p0.getMaxSupport())

    def poleFun(self, x0):
        """
        The function that determines the case of convergence
        """
        p0, beta, q = self.getAll()

        return q.integrate(
            lambda x: beta / (1 - x / x0),
            interval=[0, q.getMaxSupport() - 0.001],
            epsilon=0.001,
        )

    def animationBeta(self, path: str = "resultats/convergenceBeta.gif"):
        """
        Create an animation of the difference of the theorical limit and the
        algorithm limit according to the beta parameter.

        Parameters
        ----------
        path : str, optional
            The path of the gif save.
            The default is "resultats/convergenceBeta.gif".
        """
        # create the gif
        self.gifPrevention("GIF")

        # the different values of beta
        betas = np.arange(0.1, 1, 0.01)
        # get the maximum of support of p0
        eta0 = self.getP0().getMaxSupport()
        # the number of iteration for the convergence of the algorithm
        N = 100
        # save the beta to reset it at the and
        beta = self.getBeta()
        for b in betas:
            b = round(b, 2)
            # set the new parameter
            self.setBeta(b)
            # get the limit and plot
            self.getLimit().plot(label="limit", color="red")
            # get the algorithm limit
            calculatedLimit = self.convergence(N)
            # save the figure
            calculatedLimit.save(
                f"GIF//{b}.png",
                label=f"$\\beta$ = {b}\n$p_n(\\eta_0) = $"
                + f"{round(calculatedLimit.function(eta0), 2)}",
                color="green",
                loc="upper left",
                interval=self.getInterval(0.025),
            )
        # get all the files that have been saved
        filenames = [f"GIF//{round(b,2)}.png" for b in betas]
        # create the gif
        self.gifCreation(path, filenames, True)
        # set the value of beta with the initial value
        self.setBeta(beta)

    def animationEta_q(self, path: str = "resultats/convergenceEta_q.gif"):
        """
        Create an animation gif with different values of eta q.

        Parameters
        ----------
        path : str, optional
            The path of the gif save.
            The default is "resultats/convergenceEta_q.gif".
        """
        # create the folder
        self.gifPrevention("GIF")

        # all the different values of eta
        etas = np.arange(0.1, self.getP0().getMaxSupport() + 0.1, 0.1)
        # number of iteration for the algorithm
        N = 100
        for e in etas:
            e = round(e, 2)
            # create the measure depending on eta q
            q = Measure(lambda x: 1 / (x + 0.01) * (x <= e))
            # normalize
            q = q / q.integrate()
            # set the value of q
            self.setQ(q)
            # get the limit and plot
            self.getLimit().plot(label="limit", color="red")
            # get the algorithm limit
            calculatedLimit = self.convergence(N)
            # save the plot
            calculatedLimit.save(
                f"GIF//{e}.png",
                label=f"$\\eta_q$ = {e}",
                color="green",
                loc="upper left",
                interval=self.getInterval(0.025),
            )
        # create the file names list
        filenames = [f"GIF//{round(e,2)}.png" for e in etas]
        # create the gif
        self.gifCreation(path, filenames, True)

    def gammaRepartitionPlot(self, alpha: float, a: float, N: int = 100):
        # get the parameters
        p0, beta, q = self.getAll()
        gammaMeasure = Measure(
            lambda x: x ** (alpha - 1) * np.exp(-x), interval=[0, a]
        ).normalize()

        self.gifPrevention()
        epsilon = 1e-1
        x, limit = gammaMeasure.repartition(epsilon)
        print(x, limit)
        size = len(x)

        plot(x, limit, label="limit", color="red")

        def iterativeSave(measure, i):
            if i > a:
                newMeasure = Measure(lambda x: measure(1 - x / i), [0, a]).normalize()
                newMeasure.plot(color="green", label="density new Measure")
                print(newMeasure.integrate())

                x, repartitionPn = newMeasure.repartition(epsilon)
                print(x, repartitionPn)

                """ repartitionPn = np.zeros(size)
                for xk in range(size - 1):
                    repartitionPn[xk + 1] = repartitionPn[xk] + epsilon * measure(
                        1 - x[xk] / i
                    )
                repartitionPn /= measure.integrate(interval=[1 - a / i, 1]) * i """

                """ measure.plot(label="measure")
                axvline(1 - 2 / 90) """
                plot(x, repartitionPn, label=r"$p_{" + str(i) + "}$")
                xlabel(f"x between 0 and {a}")
                title("Repartition function of the $p_n$ and the gamma law.")
                legend()
                # savefig(f"GIF/{i}.png")
                # close()

        self.convergenceWithFunction(iterativeSave, N, 90)
        savefig("fffff.png")
        filenames = [f"GIF//{i}.png" for i in range(a + 1, N)]
        # create the gif
        # self.gifCreation("figures/gamma_repartition.gif", filenames, True)

    def gammaDensityPlot(self, alpha, a, N=100):
        gammaMeasure = Measure(
            lambda x: x ** (alpha - 1) * np.exp(-x), interval=[0, a]
        ).normalize()
        self.gifPrevention()

        def iterativeSave(measure, i):
            if i > a:
                µ = Measure(lambda x: measure(1 - x / i), interval=[0, a]).normalize()
                gammaMeasure.plot(label="Gamma measure", color="red")
                xlabel(f"x between 0 and {a}")
                title("Density of $p_n$ compared of density of Gamma")
                µ.save(f"GIF/{i}.png", label=r"$p_{" + str(i) + "}$", color="green")

        self.convergenceWithFunction(iterativeSave, N, 1)
        self.gifCreation(
            "figures/gamma_density.gif", [f"GIF/{i}.png" for i in range(a + 1, N)], True
        )


""" def TimeDifference():

    p01 = Measure(lambda x: (x <= 0.8), interval=[0, 1])
    p01 = p01/p01.integrate()
    q1 = Measure(lambda x: 1/(x+0.01)*(x <= 0.8), interval=[0, 1])
    q1 = q1/q1.integrate()
    beta = 0.56

    p02 = MeasureNul(lambda x: (x <= 0.8), interval=[0, 1])
    p02 = p02/p02.integrate()
    q2 = MeasureNul(lambda x: 1/(x+0.01)*(x <= 0.8), interval=[0, 1])
    q2 = q2/q2.integrate()

    def kingmanEq(p, beta, q):
        return ((1-beta)/p.getMean())*Id()*p+beta*q

    def kingmanEqNul(p, beta, q):
        return ((1-beta)/p.getMean())*IdNul()*p+beta*q

    kingman = Kingman(p01, beta, q1)

    kingman.timeDifferenceBetweenTwoMeasures(p01,
                                             p02,
                                             [beta, q1],
                                             [beta, q2],
                                             kingmanEq,
                                             kingmanEqNul,
                                             100,
                                             "Computation with memoization",
                                             "Computation without memoization")

 """


def animation_convergence(p0, beta, q):
    model = Kingman(p0, beta, q)
    model.animation("figures/Kingman/convergence.gif", N=100)


def animation_repartition(p0, beta, q):
    kingman = Kingman(p0, beta, q)
    kingman.animationRepartition(N=300)


def main():
    alpha = 3
    p0 = Measure(lambda x: x <= 1).normalize()
    beta = 0.1
    q = Measure(lambda x: (x <= 0.9) * alpha * (1 - x) ** (alpha - 1)).normalize()
    kingman = Kingman(p0, beta, q)
    kingman.animationRepartition(N=300)


def animation_repartition_gamma():
    alpha = 3
    p0 = OneMeasure()
    beta = 0.1
    q = Measure(lambda x: alpha * (1 - x) ** (alpha - 1)).normalize()
    h = 1e-3

    print(q.integrate(interval=[1 - h, 1], epsilon=h / 100) / (h**alpha))
    model = Kingman(p0, beta, q)
    # model.animation()
    model.gammaRepartitionPlot(alpha, 10, 100)
