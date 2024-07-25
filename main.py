import numpy as np
from matplotlib.pyplot import (
    plot,
    savefig,
    legend,
    ylim,
    xlim,
    close,
    hist,
    title,
    xlabel,
    ylabel,
    scatter,
)
import time as tea
from utils import convertTime

from measures.measures import Dirac, Measure, OneMeasure, Id
from models.Kingman import (
    animation_convergence as kingman_convergence,
    animation_repartition as kingman_repartition,
    animation_repartition_gamma as arg,
)
from models.convolution_model import (
    animation_convergence as cmodel_convergence,
    animation_repartition as cmodel_repartition,
)


if __name__ == "__main__":
    begin = tea.time()

    from models.convolution_model import Cmodel as Model
    from models.Kingman import Kingman as Model
    from models.birth_local_mutation import BirthLocalMutation
    from models.birth_kingman import BirthKingman

    """ alpha = 3
    p0 = Measure(lambda x: 1 - x).normalize()
    beta = 0.1
    q = Measure(lambda x: alpha * (1 - x) ** (alpha - 1)).normalize()
    h = 1e-3
    # kingman_convergence(p0, beta, q)
    arg() """

    """ I = [0, 10]
    beta = 1 / 2

    model = BirthLocalMutation(0.5, 1)

    model.simulation(500)
    model.anim()
    print(model.process[0]) """

    I = [0, 10]
    p0 = Measure(lambda x: x < 1, I)
    beta = 0.5
    j = Dirac(1, I)
    q = OneMeasure()

    model = BirthKingman(beta, q)
    model.simulation(500)
    model.anim("okok.gif")

    print("execution time : " + convertTime(tea.time() - begin))
