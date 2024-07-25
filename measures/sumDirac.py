import numpy as np
from matplotlib.pyplot import close, legend, plot, savefig, scatter, xlim, ylim, axvline

from measures.measures import Dirac, Measure, OneMeasure


# the class corresponding of the sum of diracs and measures
class SumDirac(Measure):

    def __init__(self, List: list, interval: list = None):
        """
        Constructor

        Parameters
        ----------
        List : list, optional
            The list of the measures in the sum.
        """
        # the list of all measures in the sum
        self.List = List
        # all the numbers of potentials Diracs in the sum
        self.numbers = None
        # the interval of definition of the sum
        if interval is not None:
            self.interval = interval
        else:
            self.interval = None
        self.epsilon = None
        self.cacheValues = {}
        self.minSupport = None
        self.maxSupport = None

    def __call__(self, x):
        """
        operator ()
        Used to call the function of the measure sum.
        example : Let µ a measure, we can write µ(0) to get the density at 0.

        Parameters
        ----------
        x: float or list

        Returns
        -------
        float
            returns the sum of all the values in the sum.
        """
        try:
            # we try to get the length (if x is a iterable)
            size = len(x)
            values = np.zeros(size)
            # if it is, we call the measure for each value in the list
            for i in range(size):
                values[i] = self(x[i])
            return values
        except TypeError:
            # if x is a float we check if we saved the value
            if x not in self.cacheValues:
                # if not, we compute the value using the function of the
                # measure and we save it
                self.cacheValues[x] = sum([measure(x) for measure in self.List])
            return self.cacheValues[x]

    def setInterval(self, interval: list):
        """
        set the interval where the sum is defined

        Parameters
        ----------
        interval : list
            The interval of definition.
        """
        for k in range(len(self.List)):
            # change all the intervals
            self.List[k].setInterval(interval)

    def getInterval(self):
        """
        Get the interval of definition of the measure sum.

        Returns
        -------
        list
            The interval of definition of the sum.
        """
        # define the min and the max of the interval by taking
        # the first interval
        minInterval = self.List[0].interval[0]
        maxInterval = self.List[0].interval[1]
        # we look every measure in the sum
        for measure in self.List:
            # get the intersection of all the intervals
            if measure.interval[0] > minInterval:
                minInterval = measure.interval[0]
            if measure.interval[1] < minInterval:
                maxInterval = measure.interval[1]
        # return the interval
        return [minInterval, maxInterval]

    def isDirac(self):
        for measure in self.List:
            if not isinstance(measure, Dirac):
                return False
        return True

    def getDiracNumbers(self):
        """
        Get all the numbers of diracs in the sum.

        Returns
        -------
        list
        """
        # if the numbers have been calculated
        if self.numbers is not None:
            return self.numbers
        else:
            numbers = []
            for measure in self.List:
                # if the measure in the list is a dirac
                if isinstance(measure, Dirac):
                    # get its number
                    numbers.append(measure.number)
            # return the numbers
            self.numbers = numbers
            return self.numbers

    def integrate(self, f=lambda x: 1, interval: list = None, epsilon: float = 0.001):
        """
        Integrate a function by the self Measure sum.

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
            Returns the sum of all the integrals.
        """
        selfinterval = self.getInterval()
        epsilon = 10e-4
        return sum([measure.integrate(f, interval, epsilon) for measure in self.List])

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
            If it is not specified, will plot the meausre on self.interval.
        setxlim: list, optional
            force the xlim of the plot
            If it is not specified, will take self.interval
        """
        # get the numbers and sort them
        numbers = sorted(self.getDiracNumbers())
        # dicretization of the space and adding the numbers
        epsilon = self.getEpsilon()
        x = np.sort(np.concatenate((np.arange(*self.getInterval(), epsilon), numbers)))

        # plot of the function measure
        plot(x, self(x), label=label, color=color)
        for measure in self.List:
            if isinstance(measure, Dirac):
                if inf:
                    axvline(measure.number, color=color)
                else:
                    scatter(measure.number, self(measure.number), color=color)
        # crop on the interval
        if setxlim is not None:
            xlim(*setxlim)
        else:
            xlim(self.getInterval())
        ylim(0, None)

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
        inf: bool = False,
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
        self.plot(
            label=label,
            color=color,
            loc=loc,
            interval=interval,
            setxlim=setxlim,
            inf=inf,
        )
        # save the figure
        savefig(filename)
        # close the figure to do not get plot
        close()

    def mergeDiracs(self):
        indexes = {}
        for index, measure in enumerate(self.List):
            if isinstance(measure, Dirac):
                if measure.number in indexes:
                    indexes[measure.number].append(index)
                else:
                    indexes[measure.number] = [index]
            else:
                if "notDirac" in indexes:
                    indexes["notDirac"].append(index)
                else:
                    indexes["notDirac"] = [index]
        newList = []
        for indexList in indexes.values():
            sumMeasure = 0
            for indexToMerge in indexList:
                sumMeasure += self.List[indexToMerge]
            newList.append(sumMeasure)
        self.List = newList

    def __add__(self, other):
        """
        operator +
        SumDirac + SumDirac
        or
        SumDirac + Measure
        or
        SumDirac + float

        Returns
        -------
        SumDirac
        """
        newList = self.List.copy()
        if isinstance(other, SumDirac):
            # combine the two lists
            newList = np.concatenate((newList, other.List.copy()))
            µ = SumDirac(newList)
            µ.mergeDiracs()
            return µ
        elif isinstance(other, Measure):
            # add the new measure to the list
            newList.append(other)
            µ = SumDirac(newList)
            µ.mergeDiracs()
            return µ
        else:
            if other == 0:
                return self
            # create the measure associated to the float
            newList.append(OneMeasure() * other)
        return SumDirac(newList)

    def __radd__(self, other):
        """
        operator +
        SumDirac + SumDirac
        or
        Measure + SumDirac
        or
        float + SumDirac

        Returns
        -------
        SumDirac
        """
        return self.__add__(other)

    def __sub__(self, other):
        """
        operator -
        SumDirac - SumDirac
        or
        SumDirac - Measure
        or
        SumDirac - float

        Returns
        -------
        SumDirac
        """
        return self.__add__(-1 * other)

    def __rsub__(self, other):
        """
        operator -
        SumDirac - SumDirac
        or
        Measure - SumDirac
        or
        float - SumDirac

        Returns
        -------
        SumDirac
        """
        newList = []
        for k in range(len(self.List)):
            newList.append(-1 * self.List[k])
        return SumDirac(newList) + other

    def __mul__(self, other):
        """
        operator *
        SumDirac * Measure
        or
        SumDirac * float

        Returns
        -------
        SumDirac
        """
        newList = []
        for k in range(len(self.List)):
            newList.append(self.List[k] * other)
        return SumDirac(newList)

    def __rmul__(self, other):
        """
        operator *
        Measure * SumDirac
        or
        float * SumDirac

        Returns
        -------
        SumDirac
        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        operator /
        SumDirac / Measure
        or
        SumDirac / float

        Returns
        -------
        SumDirac
        """
        newList = []
        for k in range(len(self.List)):
            newList.append(self.List[k] / other)
        return SumDirac(newList)

    """ def __rtruediv__(self, other):
        newList = []
        for k in range(len(self.List)):
            newList.append(other / self.List[k])
        return SumDirac(newList) """

    def exp(self):
        newList = []
        for k in range(len(self.List)):
            newList.append(self.List[k].exp())
        return SumDirac(newList)

    def abs(self):
        newList = []
        for k in range(len(self.List)):
            newList.append(self.List[k].abs())
        return SumDirac(newList)

    def __and__(self, other):
        """
        operator &
        Defined as the convolution of the measures.

        Returns
        -------
        SumDirac or Measure
            p & q returns the measure corresponding to the convolution
            of the two functions.
        """
        newList = []
        for k in range(len(self.List)):
            newList.append(other & self.List[k])
        # returns the sum of the convolutions
        return sum(newList)

    def random(self, N: int = 1):
        if N == 1:
            chooseMeasure = np.random.random()
            sumIntegral = 0
            for k in range(len(self.List)):
                sumIntegral += self.List[k].integrate()
                if chooseMeasure <= sumIntegral:
                    return self.List[k].random()
        else:
            return np.array([self.random(1) for _ in range(N)])
