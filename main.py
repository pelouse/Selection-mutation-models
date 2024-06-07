import numpy as np
from matplotlib.pyplot import plot, savefig

from measures.measures import Dirac, Measure
from models.convolution_model import main as run_cmodel

if __name__ == "__main__":
    run_cmodel()
