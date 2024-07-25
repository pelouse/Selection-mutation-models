import numpy as np
from matplotlib.pyplot import axvline, close, legend, plot, savefig, scatter, xlim, ylim
from numpy import random
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde


class Measure:

    def __init__(self, funOrList, interval: list = [0, 1], precision: int = 4):
        """
        Constructor

        Parameters
        ----------
        funOrList : function or list
            If it is a function, the measure is defined as it.
            If it is a list of values, it will compute the density.
        interval : list of 2 values, optional
            The defintition interval of the measure.
            The first value has to be lower than the second.
            The default is [0, 1].
        precision : int, optional
            The precision to call the measure density.
            If the precision is 4, the function will be call with precision 10e-4.
            That means when you want to compute the density value at 0,5001, it will
            return the density value at 0,500.
            The default is 4.
        """
        # attributes
        # the function of the measure
        self.function = None
        # the mean of the measure
        self.mean = None
        # the interval of definition
        self.interval = interval
        # the precision when calling the measure density
        self.precision = precision
        self.epsilon = None
        self.minSupport = None
        # the support of the measure
        self.maxSupport = None
        # the values that have been calculated
        self.cacheValues = {}

        # if funOrList is a list a random values
        if isinstance(funOrList, list) or isinstance(funOrList, np.ndarray):
            colinearity = True
            first_element = funOrList[0]
            for element in funOrList:
                if element != first_element:
                    colinearity = False
                    break
            if colinearity:
                self.function = lambda x: x == first_element * len(funOrList)
            else:
                # compute of the density
                print(len(funOrList))
                self.function = np.vectorize(lambda x: gaussian_kde(funOrList)(x)[0])
        # if funOrList is a function
        else:
            # Definition of the measure as the function
            self.function = funOrList

    def __call__(self, x: float):
        """
        operator ()
        Used to call the function of the measure.
        example : Let µ a measure, we can write µ(0) to get the density at 0.
        It will compute the density at x rounded at 10e-n with n the precision of the measure.

        Parameters
        ----------
        x: float or list

        Returns
        -------
        float
        """
        if isinstance(x, list) or isinstance(x, np.ndarray):
            # we try to get the length (if x is a iterable)
            size = len(x)
            values = np.zeros(size)
            # if it is, we call the measure for each value in the list
            for i in range(size):
                values[i] = self(x[i])
            return values
        else:
            # round the value of x
            x = round(x, self.precision)
            # if x is a float we check if we saved the value
            if x not in self.cacheValues:
                # if not, we compute the value using the function of the
                # measure and we save it
                if self.interval[0] <= x and x <= self.interval[1]:
                    self.cacheValues[x] = self.function(x)
                else:
                    self.cacheValues[x] = 0
            return self.cacheValues[x]

    def __eq__(self, other):
        """
        operator ==
        Measure + Measure

        Returns
        -------
        boolean
            Returns True if the two measure are almost equals.
            Returns false otherwise.
        """
        # if the intervals are not equal, the measures are considered not equal
        if self.interval != other.interval:
            return False
        else:
            # discretization of the interval
            x = np.arange(*self.interval, 0.01)
            # the error to not exceed
            error = 10e-4
            # travel the interval
            for k in x:
                # update the error
                error -= self(k) - other(k)
            # if the error is sufficiently low, they are considered equal
            return error >= 0

    def __neg__(self):
        """
        operator -Measure

        Returns
        -------
        Meausre
            Returns the opposite of the measure.
        """
        return Measure(lambda x: -self(x), self.interval)

    def __add__(self, other):
        """
        operator +
        Measure + Measure
        or
        Measure + number

        Returns
        -------
        Measure
        """
        # if we sum two measures
        if isinstance(other, Measure):
            # create a new measure
            # which the function is the sum of the two functions
            return Measure(
                lambda x: self(x) + other(x),
                [
                    max(self.interval[0], other.interval[0]),
                    min(self.interval[1], other.interval[1]),
                ],
            )
        # if we sum a measure and a number
        else:
            # return the sum
            return Measure(lambda x: self(x) + other, self.interval)

    def __radd__(self, other):
        """
        operator +
        number + Measure

        Returns
        -------
        Measure
        """
        return self.__add__(other)

    def __sub__(self, other):
        """
        operator -
        Measure - Measure
        or
        Measure - number

        Returns
        -------
        Measure
        """
        # take the opposite of the second measure and sum.
        return self.__add__(-other)

    def __rsub__(self, other):
        """
        operator -
        number - Measure

        Returns
        -------
        Measure
        """
        return other + -self

    def __mul__(self, other):
        """
        operator *
        Measure * Measure
        or
        Measure * float

        Returns
        -------
        Measure
        """
        from measures.measures import Dirac

        # if we multiply two measures
        if isinstance(other, Measure):
            if isinstance(other, Dirac):
                return other.__mul__(self)
            # creation of a measure which the function
            # is the multiplication of the two measures
            return Measure(
                lambda x: self(x) * other(x),
                [
                    max(self.interval[0], other.interval[0]),
                    min(self.interval[1], other.interval[1]),
                ],
            )
        # if we multiply a measure and a number
        else:
            # the function of the new measure
            def f(x):
                return other * self(x)

            # return the measure
            return Measure(f, self.interval)

    def __rmul__(self, other):
        """
        operator *
        Measure * Measure
        or
        float * Measure

        Returns
        -------
        Measure
        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        operator /
        Measure / Measure
        or
        Measure / float

        Returns
        -------
        Measure
        """
        # if we divide two measures
        if isinstance(other, Measure):
            # creation of the new measure
            return Measure(
                lambda x: self(x) / other(x),
                [
                    max(self.interval[0], other.interval[0]),
                    min(self.interval[1], other.interval[1]),
                ],
            )
        # if we divide a measure by a number
        else:
            # creation of the function
            def f(x):
                return self(x) / other

            # returns the new measure
            return Measure(f, self.interval)

    def __rtruediv__(self, other):
        """
        operator /
        Measure / Measure
        or
        float / Measure

        Returns
        -------
        Measure
        """
        # if we divide two measures
        if isinstance(other, Measure):
            # creation of the new measure
            return Measure(
                lambda x: other(x) / self(x),
                [
                    max(self.interval[0], other.interval[0]),
                    min(self.interval[1], other.interval[1]),
                ],
            )
        # if we divide a number by a measure
        else:
            # creation of the function
            def f(x):
                return other / self(x)

            # returns the new measure
            return Measure(f, self.interval)

    def __pow__(self, other):
        """
        operator **
        Measure ** float

        Returns
        -------
        Measure
        """
        return Measure(lambda x: self(x) ** other, self.interval)

    def exp(self):
        """
        function for the exponential of a measure

        Returns
        -------
        Measure
            returns exp(Measure).
        """
        return Measure(lambda x: np.exp(self(x)), self.getInterval())

    def abs(self):
        return Measure(lambda x: np.abs(self(x)), self.getInterval())

    def copy(self, precision: float = 0.001):
        """
        copy the measure by interpolate the values of the function.

        Returns
        -------
        Measure
            returns a copy of the measure.
        """
        # compute some values of the measure
        x = np.arange(self.interval[0], self.interval[1], precision)
        y = self(x)

        # interpolation of the values calculated
        newf = interp1d(x, y)

        # the function to return that returns 0 if not in the interval
        def r(x):
            if self.interval[0] <= x and x <= self.interval[1]:
                return newf(x)
            else:
                return 0

        # create and return of the new measure
        return Measure(np.vectorize(r), self.interval.copy())

    def integrate(self, f=lambda x: 1, interval: list = None, epsilon: float = None):
        """
        Integrate a function by the self Measure.

        Parameters
        ----------
        f : function, optional
            The function to integrate.
            The default is the one function.
        interval : list, optional
            The interval to integrate the function f by the self Measure.
            If not specified, will use the self.interval.
        epsilon : float, optional
            The precision to compute the integral approximation.
            The default is 0.001.

        Returns
        -------
        float
            The integral of the function by the measure.
        """
        # choose the interval to integrate
        if interval is None:
            interval = self.interval
        integral = 0
        if epsilon is None:
            epsilon = self.getEpsilon()
        # discretization of the interval
        x = np.arange(interval[0], interval[1] + epsilon, epsilon)
        # algorithm
        for xk in x:
            # compute the integral
            integral += epsilon * f(xk) * self(xk)
        return integral

    def repartition(self, epsilon=None):
        """
        Repartition function of the measure

        Parameters
        ----------
        epsilon: float, optionnal
            The precision to compute the integral approximation.
            The default is 0.001.

        Returns
        ------
        x: List
            The discretization of the measure's interval
        repartitionList: List
            The list of values corresponding to the function
            x |-> integral between min(interval) and x of the measure.
        The two lists are the same dimensions and are ready to be plot.
        """
        from measures.measures import SumDirac, Dirac

        listDirac = []
        if isinstance(self, SumDirac):
            for measure in self.List:
                if isinstance(measure, Dirac):
                    listDirac.append(measure.number)
        if epsilon is None:
            epsilon = self.getEpsilon()
        # discretization of the interval
        # we start before the min of the interval to have 0
        x = np.arange(
            self.getInterval()[0] - epsilon, self.getInterval()[1] + epsilon, epsilon
        )
        # get the length of the discretization
        size = len(x)
        # initialize the return list
        repartitionList = np.zeros(size)
        # travel the interval
        for xk in range(size - 1):
            if len(listDirac) > 0 and x[xk + 1] >= listDirac[0]:
                repartitionList[xk + 1] = repartitionList[xk] + self(listDirac[0])
                listDirac.pop(0)
            else:
                # compute the integral between min(interval) and xk
                repartitionList[xk + 1] = repartitionList[xk] + epsilon * self(x[xk])
        return x, repartitionList

    def __and__(self, other):
        """
        operator &
        Defined as the convolution of the measures.

        Returns
        -------
        Measure
            p & q returns the measure corresponding to the convolution
            of the two functions.
        """
        from measures.measures import Dirac

        if isinstance(other, Dirac) and isinstance(self, Dirac):
            return Dirac(
                self.number + other.number,
                self.interval,
                lambda x: self.function(self.number) * other.function(x - self.number),
            )

        # if the second measure is a Dirac
        if isinstance(other, Dirac):
            # just modify the measure function
            return Measure(
                lambda x: other.function(other.number) * self(x - other.number),
                self.interval,
            )
        # if self is a Dirac measure
        if isinstance(self, Dirac):
            # just modify the second measure function
            return Measure(
                lambda x: self.function(self.number) * other(x - self.number),
                other.interval,
            )

        # defintion of the convolution
        def convolution(f, g, x):
            return g.integrate(lambda t: f(x - t))

        # return the measure with the intersection of the intervals
        return Measure(
            lambda x: convolution(self, other, x),
            [
                max(self.interval[0], other.interval[0]),
                min(self.interval[1], other.interval[1]),
            ],
        )

    def getMean(self, epsilon: float = 0.001):
        """
        Get the mean of the measure.

        Parameters
        ----------
        epsilon : float, optional
            The precision to compute the integral approximation.
            The default is 0.001.

        Returns
        -------
        float
            returns the integral of the function identity.
        """
        epsilon = 10e-4
        # Compute of the mean if it has never been calculated
        if self.mean is None:
            self.mean = self.integrate(lambda x: x, epsilon=epsilon)
        return self.mean

    def getEpsilon(self):
        if self.epsilon is None:
            self.epsilon = 1e-3 * 10 ** (int(np.log10(self.intervalRange())))
        return self.epsilon

    def getMinSupport(self, precision: float = 1e-3):
        """
        Get the minimum of the support of a measure.

        Returns
        -------
        float
            Return the greatest value such that all lower values give 0 by
            the measure.
        """
        # if the value has been calculated
        if self.minSupport is not None:
            return self.minSupport
        else:
            # discretization of the interval upside down
            x = np.arange(self.getInterval()[0], self.getInterval()[1], precision)
            # we going to the lowest value
            for xk in x:
                # print(xk)
                # check if the values are different of zero
                if self(xk) != 0:
                    # get and return the value rounded
                    self.minSupport = round(xk, 3)
                    return self.minSupport
            # if it is the null function, get the last value
            self.minSupport = self.getInterval()[1]
            return self.minSupport

    def getMaxSupport(self, precision: float = 1e-3):
        """
        Get the maximum of the support of a measure.

        Returns
        -------
        float
            Return the lowest value such that all greater values give 0 by
            the measure.
        """
        # if the value has been calculated
        if self.maxSupport is not None:
            return self.maxSupport
        else:
            # discretization of the interval upside down
            x = np.arange(self.interval[1], self.interval[0], -precision)
            # we going to the lowest value
            for k in range(len(x)):
                # check if the values are different of zero
                if self(x[k]) != 0:
                    # get and return the value rounded
                    self.maxSupport = round(x[k], 3)
                    return self.maxSupport
            # if it is the null function, get the first value
            self.maxSupport = self.interval[0]
            return self.maxSupport

    def setInterval(self, interval: list):
        self.interval = interval
        self.epsilon = None

    def getInterval(self):
        return self.interval

    def intervalRange(self):
        interval = self.getInterval()
        return interval[1] - interval[0]

    def isDirac(self):
        return False

    def isProbability(self, epsilon: float = 0.001):
        """
        Condition of the measure to be a probability measure

        Parameters
        ----------
        epsilon : float, optional
            The precision to compute the integral approximation.
            The default is 0.001.

        Returns
        -------
        boolean
            returns True if the measure is a probability measure.
            False otherwise.
        """
        epsilon = self.getEpsilon()
        # Calculate the difference between 1 and the integration of the measure
        # if the integral is near to one, returns True. False otherwise.
        return np.abs(self.integrate(epsilon=epsilon) - 1) < 0.1

    def normalize(self):
        """
        Normalize the measure to get a probability measure
        """
        integral = self.integrate()
        if integral == 0:
            raise Exception(
                "Can not normalize the measure because the integral is zero."
            )
        else:
            return self.__truediv__(integral)

    def plot(
        self,
        label: str = None,
        color: str = None,
        loc: str = "best",
        interval: list = None,
        setxlim: list = None,
        inf: bool = False,
    ):
        """
        plot the measure on the interval.

        Parameters
        ----------
        label : string, optional
            The name of the density
        color : string, optional
            The color of the plot
        loc : string, optional
            The localisation of the label on the plot.
            The default value is the default value of matplotlib.
        interval: list, optional
            prevent the measure to be plot outside interval.
            If it is not specified, will plot the measure on self.interval.
        setxlim: list, optional
            force the xlim of the plot
            If it is not specified, will take self.interval
        """
        # put lines to delimitate the interval of the measure
        if interval is not None:
            # if the first value is different
            if interval[0] != self.getInterval()[0]:
                # we add the line
                axvline(interval[0], color="goldenrod", linestyle="--")
            # if the second value is different
            if interval[1] != self.getInterval()[1]:
                # we add the line
                axvline(interval[1], color="goldenrod", linestyle="--")
        # if interval is not specified, we just get the interval of the measure
        else:
            interval = self.getInterval()
        # discretization of the interval
        epsilon = self.getEpsilon()
        x = np.arange(self.getInterval()[0], self.getInterval()[1] + epsilon, epsilon)
        # compute the measure
        y = self(x)
        # plot of the function measure
        plot(x, y, label=label, color=color)
        # ylim bottom is the minimum of the function or 0
        # ylim top is the maximum of the function or the previous ylim
        previous_ylim = ylim()
        ylim(min(min(y), 0, previous_ylim[0]), max(max(y), previous_ylim[1]))
        # crop on the interval
        if setxlim is not None:
            xlim(setxlim)
        else:
            xlim(self.interval)
        # print the legend
        if label is not None:
            legend(loc=loc)

    def save(
        self,
        filename: str = "measure.png",
        label: str = None,
        color: str = None,
        loc: str = "best",
        interval: list = None,
        setxlim: list = None,
    ):
        """
        Save the figure

        Parameters
        ----------
        filename : string
            The path of the file to be save.
        label : string, optional
            The name of the density
        color : string, optional
            The color of the plot
        loc : string, optional
            The localisation of the legend.
            The different strings are :
                'upper left', 'upper right', 'lower left', 'lower right',
                'upper center', 'lower center', 'center left', 'center right',
                'center', 'best'
            The default is 'best'.
        interval: list, optional
            prevent the measure to be plot outside interval.
            If it is not specified, will plot the meausre on self.interval.
        setxlim: list, optional
            force the xlim of the plot
            If it is not specified, will take self.interval
        """
        # plot the measure
        self.plot(label=label, color=color, loc=loc, interval=interval, setxlim=setxlim)
        # save the figure
        savefig(filename)
        # close the figure to do not get plot
        close()

    def random(self, N: int = 1):
        """
        Simulate a random variable according to the density.

        Parameters
        ----------
        N : int, optional
            The number of variable to simulate. The default is 1.

        Returns
        -------
        int if N == 1, list if N > 1.
            The variable or list of random variables according to the density.
        """
        if N == 1:
            # rejection method
            while True:
                # simulation of two variables
                # first one is the variable on the interval to return
                variable = random.uniform(*self.interval)
                # second one is the rejection variable in [0, 1]
                rejection = random.random()
                # if the rejection variable is under the density
                if rejection <= self.function(variable):
                    # return the corresponding variable
                    return variable
        # if N > 1
        else:
            # creating a list of variables
            return np.array([self.random(1) for _ in range(N)])
