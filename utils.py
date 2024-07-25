import os

import imageio.v2 as imageio
import numpy as np
from numpy import random

from measures.measures import Measure


def zeroBijection(function, interval=[0, 1], epsilon=0.001):
    """
    get an approximation of the zero of a acreasing function

    Parameters
    ----------
    function : function
        The function to compute the zero.
    interval : list, optional
        The interval to check. The default is [0, 1].
    epsilon : float, optional
        The precision of the return value. The default is 0.001.

    Returns
    -------
    float
        The x such as function(x) is near 0.
    """
    # create the list of tested values
    values = np.arange(*interval, epsilon)
    # if we start over 0, we stop the program
    if function(values[0]) > 0:
        # stop the program
        return interval[0]
        raise Exception("Start of the ckeck over zero")
    for x in values:
        # if we pass over 0
        if function(x) >= 0:
            # return of the first value such as function(x) >= 0
            return x
    # stop the program
    raise Exception("No zero found on the interval")


def convertTime(seconds):
    seconds = int(seconds)
    minutes = int(seconds / 60)
    seconds -= 60 * minutes
    hours = int(minutes / 60)
    minutes -= 60 * hours
    days = int(hours / 24)
    hours -= 24 * days
    string = ""
    if days != 0:
        string += str(days) + " days, " + str(hours) + " h, " + str(minutes) + " min, "
    elif hours != 0:
        string += str(hours) + " h, " + str(minutes) + " min, "
    elif minutes != 0:
        string += str(minutes) + " min, "
    string += str(seconds) + " s"
    return string


#####################################################################
#                                                                   #
#                      RANDOM THINGS                                #
#                                                                   #
#####################################################################


def find(tab):
    L = len(tab)
    firstFill = None
    for k in range(L - 1):
        if tab[k] != 0 and firstFill is None:
            firstFill = k
        if tab[k] != 0 and tab[k] >= tab[k + 1]:
            return k
    return firstFill


def aaa(positions, quantity):
    s = [quantity if k == 0 else 0 for k in range(positions)]
    r = [s.copy()]
    while s[-1] != quantity:
        index = find(s)
        s[index] -= 1
        s[index + 1] += 1
        r.append(s.copy())
    return r


def transcription(tab):
    r = []
    L = len(tab)
    for i in range(L):
        for j in range(tab[i]):
            r.append((i) / (L - 1))
    return r


def ani(arange, filename):
    # create a folder GIF if it does not exists
    if not os.path.exists("GIF"):
        os.makedirs("GIF")

    # creation of the function into the algorithm
    N = len(arange)
    y = arange[0][0]
    filenames = []
    µ1 = Measure(arange[0])
    µ1 = µ1 / µ1.integrate()
    for i in range(N):
        µ = Measure(arange[i])
        µ = µ / µ.integrate()
        µ.save(
            f"GIF//{i}.png",
            color="green",
        )

        # Measure(arange[i]).plot()
    # get every file name that have been saved
    filenames = [f"GIF//{i}.png" for i in range(N)]
    # creating the gif
    with imageio.get_writer(filename, mode="I", duration=3, loop=not True) as writer:
        for filename in filenames:
            # read the picture
            image = imageio.imread(filename)
            # add the picture to the gif
            writer.append_data(image)
            # delete the picture
            os.remove(filename)


# a = random.normal(0, 1, 100000)
# b = []
# for k in a:
#     if k >= 0 and k <= 1:
#         b.append(k)
