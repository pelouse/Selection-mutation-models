import numpy as np
from matplotlib.pyplot import axvline, close, legend, plot, savefig, scatter, xlim, ylim
from numpy import random
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

from measures.measures import Measure, OneMeasure


# Dirac measure
class Dirac(Measure):

    def __init__(
        self, number: float, interval: list = [0, 1], function=lambda x: 1 + 0 * x
    ):
        """
        Constructor

        Parameters
        ----------
        number : float
            The number where the dirac will be 1.
        interval : list, optional
            The interval of the measure. The default is [0, 1].
        function : function, optional
            The function in front of the Dirac.
            The default is lambda x: 1 + 0*x.
        """
        # call the mothers constructor
        super().__init__(lambda x: function(x) * (x == number), interval=interval)
        # save the number of the dirac
        self.number = number
        # the function in front of the dirac (f*dirac)
        self.f = function

    def __call__(self, x):
        """
        operator ()
        Used to call the function of the measure.
        example : Let µ a measure, we can write µ(0) to get the density at 0.

        Parameters
        ----------
        x: float or list

        Returns
        -------
        float
        0 if x != diracs number or f(x) if dirac = 1
        """
        try:
            size = len(x)
            values = np.zeros(size)
            for k in range(size):
                values[k] = self(x[k])
            return values
        except:
            if x == self.number:
                return self.f(x) * (self.interval[0] <= x and x <= self.interval[1])
            else:
                return 0

    def plot(
        self,
        label: str = None,
        color: str = None,
        loc: str = "best",
        interval: list = None,
        setxlim: list = None,
    ):
        """
        Plot the dirac density.

        Parameters
        ----------
        label : string, optional
            The name of the curve to plot.
            If not specified, do not print the legend
        color : string, optional
            The color of the curve.
            If not specified, the different curves will have different colors.
        loc : string, optional
            The localisation of the legend.
            The different strings are :
                'upper left', 'upper right', 'lower left', 'lower right',
                'upper center', 'lower center', 'center left', 'center right',
                'center', 'best'
            The default is 'best'.
        interval : list, optional
            The interval to plot the measure. Wil put yellow lines to show the
            interval. If not specified, will plot on the whole self.interval.
        setxlim: list, optional
            force the xlim of the plot
            If it is not specified, will take self.interval
        """
        # put lines to delimitate the interval of the measure
        if interval is not None:
            # if the first value is different
            if interval[0] != self.interval[0]:
                # we add the line
                axvline(interval[0], color="goldenrod", linestyle="--")
            # if the second value is different
            if interval[1] != self.interval[1]:
                # we add the line
                axvline(interval[1], color="goldenrod", linestyle="--")
        # if interval is not specified, we just get the interval of the measure
        else:
            interval = self.interval
        # discretization of the interval
        x = np.concatenate(
            (
                np.arange(self.interval[0], self.number, 0.001),
                np.arange(self.number, self.interval[1], 0.001),
            )
        )
        y = self(x)
        # plot of the function measure
        plot(x, y, label=label, color=color)
        scatter(self.number, self(self.number), color=color, label=label)
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

    def integrate(self, f=lambda x: 1, interval: list = None, epsilon: float = 0.001):
        """
        Integrate a function by the self Dirac.

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
            It is not use but it correspond to the function parameters of the Measure.integrate()
            The default is 0.001.

        Returns
        -------
        float
            The integral of the function by the measure.
        """
        if interval is None:
            interval = self.interval
        # if the dirac has its not null value in the interval
        if self.number >= interval[0] and self.number <= interval[1]:
            return f(self.number) * self.f(self.number)
        else:
            return 0

    def __mul__(self, other):
        """
        operator *
        Dirac * Measure
        or
        Dirac * float

        Returns
        -------
        Dirac
        """
        # if other is a Measure
        if isinstance(other, Measure):
            # return the dirac but the function in front of is the
            # multiplicacion of the two functions
            return Dirac(
                self.number,
                [
                    max(self.interval[0], other.interval[0]),
                    min(self.interval[1], other.interval[1]),
                ],
                lambda x: self.f(x) * other(x),
            )
        else:
            # else, return a new dirac with its function changed
            return Dirac(self.number, self.interval, lambda x: self.f(x) * other)

    def __rmul__(self, other):
        """
        operator *
        Measure * Dirac
        or
        float * Dirac

        Returns
        -------
        Dirac
        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        operator /
        Dirac / Measure
        or
        Dirac / float

        Returns
        -------
        Dirac
        """
        # if other is a Measure
        if isinstance(other, Measure):
            # return a dirac with a changed value
            return Dirac(
                self.number,
                [
                    max(self.interval[0], other.interval[0]),
                    min(self.interval[1], other.interval[1]),
                ],
                lambda x: self.f(x) / other(x),
            )
        # if other is a float
        else:
            return Dirac(self.number, self.interval, lambda x: self.f(x) / other)

    def __add__(self, other):
        """
        operator +
        Dirac + Measure
        or
        Dirac + float

        Returns
        -------
        SumDirac
        """
        from measures.sumDirac import SumDirac

        # if other is a Measure
        if isinstance(other, Measure):
            # return the sum of the two measures
            return SumDirac([self, other])
        # if other is a float
        else:
            # transform other into a measure
            return SumDirac([self, OneMeasure() * other])

    def __radd__(self, other):
        """
        operator +
        Measure + Dirac
        or
        float + Dirac

        Returns
        -------
        SumDirac
        """
        return self.__add__(other)

    def __sub__(self, other):
        """
        operator -
        Dirac - Measure
        or
        Dirac - float

        Returns
        -------
        SumDirac
        """
        return self.__add__(-1 * other)

    def __rsub__(self, other):
        """
        operator -
        Measure - Dirac
        or
        float - Dirac

        Returns
        -------
        SumDirac
        """
        newD = -1 * self
        return newD + other
