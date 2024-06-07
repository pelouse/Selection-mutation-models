import os
import time

import imageio.v2 as imageio
import numpy as np
from matplotlib.pyplot import legend, plot, title, xlabel, ylabel

from measures.measures import Dirac, Measure, SumDirac


class SelectionMutation:

    def __init__(self, equation, init, params: list = None,
                 indexesOfDependent: list = []):
        """
        Constructor

        Parameters
        ----------
        equation : function
            The equation for the algorithm. The function has to be like :
                f(measure, param1, param2, ..., paramN)
        init : Measure
            The initial measure.
        params : list, optional
            The list of the parameters to calculate the function equation.
            The list corresponds to the function arguments :
                [param1, param2, ..., paramN]
        indexesOfDependent: list, optional
            The list of indexes of parameters which are dependent of the actual
            measure.
            The dependent parameters are function of the measure at any time.
            example : if = [0, 2], the first and third parameters are
            dependents of the measure.
        """
        self.equation = equation
        self.params = params
        self.init = init
        self.dependent = indexesOfDependent

    def getParam(self, index: int):
        return self.params[index]

    def getParams(self):
        return self.params

    def getInit(self):
        return self.init

    def setParam(self, newParam, index: int):
        self.params[index] = newParam

    def setParams(self, newParams: list):
        self.params = newParams

    def setInit(self, newInit: Measure):
        self.init = newInit

    def getLimit(self):
        pass

    def iteration(self, measure: Measure):
        """
        One iteration of the algorithm using the equation.

        Parameters
        ----------
        measure : Measure
            The measure that will be updated.

        Returns
        -------
        Measure
            The new measure.
        """
        # if there is no parameter
        if self.params is None:
            return self.equation(measure)
        # if there are parameters
        else:
            # get all the values of parameters
            params = []
            # we check every parameter
            for k in range(len(self.params)):
                # if the parameter is dependent
                if k in self.dependent:
                    # we take the value according to the measure
                    params.append(self.params[k](measure))
                else:
                    # else, we just take the parameter
                    params.append(self.params[k])
            # finally, we use the values in the equation
            return self.equation(measure, *params)

    def convergenceWithFunction(self, function, N: int = 100, k: int = 1):
        """
        Execution of the algorithm but every step, it will call the function.

        Parameters
        ----------
        function : function
            The function that will be called every iteration of the algorithm.
            The function must be like :
                f(measure, i) with i the iteration number
        N : int, optional
            The number of iteration. The default is 100.
        k : int, optional
            The frequency of call of the function.
            If k = 2, the function will be called every two iterations.
            The default is 1.

        Returns
        -------
        measure : Measure
            The final measure.
        """
        # get a copy of the initial measure
        measure = self.init
        # the loop of the algorithm
        for i in range(N):
            # the condition to call the function
            if i % k == 0:
                # the function call
                function(measure, i)
            # update the measure
            measure = self.iteration(measure)
        return measure

    def convergence(self, N: int = 100):
        """
        The simple algorithm of convergence.

        Parameters
        ----------
        N : int, optional
            Number of iterations. The default is 100.

        Returns
        -------
        Measure
            The algorithm will converges into this measure.
        """
        # definition of a function that do nothing
        def nothing(measure, i):
            pass
        # call the algorithm but doing nothing
        return self.convergenceWithFunction(nothing, N=N, k=N + 1)

    def plotConvergence(self, N: int = 100, k: int = 10):
        """
        Plot the convergence of the model

        Parameters
        ----------
        N : int, optional
            Number of iteration for the algorithm. The default is 100.
        k : int, optional
            The frequency of call of the function.
            If k = 2, the function will be called every two iterations.
            The default is 10.
        """
        # the function that will be called during the algorithm
        def iterativePlot(measure, i):
            # ploting the actual measure
            measure.plot("iteration " + str(i))
        # calling the algorithm
        measure = self.convergenceWithFunction(iterativePlot, N=N, k=k)
        # calling the function one last time
        measure.plot("iteration " + str(N))

    def gifPrevention(self, folderPath: str = "GIF"):
        # create a folder GIF if it does not exists
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)

    def gifCreation(self, path: str, filenames: list, loop: bool):
        # creation of the gif
        with imageio.get_writer(path, mode='I', duration=5,
                                loop=not loop) as writer:
            for filename in filenames:
                # read the picture
                image = imageio.imread(filename)
                # add the picture to the gif
                writer.append_data(image)
                # delete the picture
                os.remove(filename)

    def animation(self,
                  filename: str = "mygif.gif",
                  N: int = 100,
                  loop: bool = True,
                  interval: list = None,
                  diracs = []):
        """
        Create an animation gif to see the convergence of the measures.

        Parameters
        ----------
        filename : str
            The path to save the gif.
        N : int, optional
            The number of iteration for the algorithm and the number of frame
            in the gif.
            The default is 100.
        loop : bool, optional
            Is the gif doing a loop.
            The value is True if yes and False if no.
            The default is True.
        interval: list, optional
            prevent the model to be plot outside interval.
            If it is not specified, will plot the meausre on the whole interval
        """
        # create the folder if not exists
        self.gifPrevention("GIF")

        limit = self.getLimit()
        renormalize = []
        if isinstance(limit, SumDirac):
            for measure in limit.List:
                if isinstance(measure, Dirac):
                    renormalize.append(measure.number)
                else:
                    a = measure.integrate()
        renormalize.sort()
        # creation of the function into the algorithm
        def iterativeSave(measure, i):
            if limit is not None:
                # plot the limit if specified
                limit.plot(color="red", label="limit")
                if len(renormalize) != 0:
                    """ def fun(x):
                        if x in renormalize:
                            return measure(x)/sum(measure(renormalize))
                        else:
                            return measure(x)
                    f = np.vectorize(fun)
                    µ = Measure(f, measure.interval) """
                    
                    autour = (measure.interval[1]-measure.interval[0])/10
                    # a = sum(measure(renormalize))
                    a = 0
                    a += measure.integrate(interval=[0, renormalize[0] - autour])
                    rdown = np.array(renormalize) + autour
                    rup = np.array(renormalize) - autour
                    for k in range(1, len(renormalize)):
                        a += measure.integrate(interval=[rdown[k-1], rup[k]])
                    def fun(x):
                        for k in range(len(renormalize)):
                            if rup[k] <= x <= rdown[k]:
                                return max(measure(x)/sum(measure(renormalize)) - a,0)
                        return measure(x)
                    f = np.vectorize(fun)
                    µ = Measure(f, measure.interval)
                    # µ = measure/sum(measure(renormalize))
                else:
                    µ = measure
            epsilon = measure.getEpsilon()
            # saving the measure's plot
            µ.save(filename=f'GIF//{i}.png',
                         label=f'iteration {i}',
                         color="green",
                         loc="upper left",
                         interval=interval,
                         setxlim=[measure.interval[0]-epsilon, measure.interval[1]+epsilon])

        # call the algorithm
        measure = self.convergenceWithFunction(iterativeSave, N=N, k=1)
        # save one last time
        iterativeSave(measure, N)

        # get every file name that have been saved
        filenames = [f'GIF//{i}.png' for i in range(N+1)]
        # creating the gif
        self.gifCreation(filename, filenames, loop)

    def animationParameter(self,
                           parameterIndex: int,
                           arange: list,
                           filename: str = "mygif.gif",
                           loop: bool = True):
        """
        Create an animation gif to see the limit measure according to a
        parameter.

        Parameters
        ----------
        parameterIndex : int
            The index of the parameter to modify. If index = 0, the first
            parameter of the parameter list we be modify using the arange list.
        arange : list
            All the value to use to change the parameter.
        filename : str
            The path to save the gif.
        loop : bool, optional
            Is the gif doing a loop.
            The value is True if yes and False if no.
            The default is True.
        """
        # create the folder if not exists
        self.gifPrevention("GIF")

        # get the number of frames in the animation
        N = len(arange)
        # we save the initial parameter to reuse it at the end
        saveParam = self.getParam(parameterIndex)
        for k in range(N):
            # change the parameter using the k-th value of arange
            self.setParam(arange[k], parameterIndex)
            # get the limit
            limit = self.getLimit()
            if limit is not None:
                # plot the limit if it can be compute
                limit.plot("limit", "red", method="infinity")
            # get the limit measure by algorithm
            measure = self.convergence()
            # save the plot
            measure.save(f'GIF//{k}.png', f'beta = {arange[k]}',
                         "green", loc='upper left')
        # at the end we change to the initial value
        self.setParam(saveParam, parameterIndex)
        # get every file name that have been saved
        filenames = [f'GIF//{i}.png' for i in range(N)]
        # creating the gif
        self.gifCreation(filename, filenames, loop)

    def timeDifferenceBetweenTwoMeasures(self,
                                         init1: Measure,
                                         init2: Measure,
                                         params1: list,
                                         params2: list,
                                         equation1,
                                         equation2,
                                         N: int,
                                         label1: str,
                                         label2: str):
        """
        Plot the curve of the time used to compute according to iteration
        number of the algorithm

        Parameters
        ----------
        init1 : Measure
            The initial Measure of the fastest measure class.
        init2 : Measure
            The initial Measure of the slowest measure class.
        params1 : list
            The parameters list of the fastest measure class.
        params2 : list
            The parameters list of the slowest measure class.
        equation1 : TYPE
            The equation of the fastest measure class.
        equation2 : TYPE
            The equation of the slowest measure class.
        N : int
            The total number of iterations.
        label1 : str
            The label to print for the fastest measure class.
        label2 : str
            The label to print for the slowest measure class.
        """
        # get the parameters of the model
        self.equation = equation1
        self.params = params1

        # get all the integers to become the xaxis plot
        n = np.arange(1, N + 1, 1)
        # the list of the different times
        # this list has increasing values
        times = np.zeros(N)
        # get the time at the begining of the algorithm
        begin = time.time()
        # do N iterations of the program
        for k in range(N):
            init1 = self.iteration(init1)
            # get the time from the beginning to now
            times[k] = time.time() - begin
        # plot the curve in green
        plot(n, times, label=label1, color="green")

        # get the parameters
        self.equation = equation2
        self.params = params2

        # the list of the different times
        # this list has increasing values
        times = np.zeros(N)
        # get the time at the begining of the algorithm
        begin = time.time()
        # do N iterations of the program
        for k in range(N):
            init2 = self.iteration(init2)
            # get the time from the beginning to now
            times[k] = time.time() - begin
        # plot the curve in red
        plot(n, times, label=label2, color="red")

        # labels and title
        xlabel("number of iterations")
        ylabel("time in seconds")
        title("Duration of the algorithm")
        legend()
