import numpy as np

##########
# Macros #
##########


types = ["Gaussian", "1/f", "1/f^2", "spectrum-equalized", "natural"]

"""
Loads a numpy dataset and return it.
"""
def load_dataset(filename):

	return np.load(filename)

def find_nearest_neighbor(patch, Np):