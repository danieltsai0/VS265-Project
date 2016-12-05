import pickle, signal, sys, util
import numpy as np
import matplotlib.pyplot as plt

from util import *

"""
Estimates entropy and relative dimension of a given dataset. 

Arguments: 
	tod: type of data being measured 
	n_neighbors_pow: used to exponentiate the number dos.
"""
def estimate(*args):

	try:
		_, _, tod, n_neighbors_pow = args
		# Load patches 
		Tp, Nps = util.load_patches(tod, n_neighbors_pow)
		# Generate distances for each target patch
		D = [[util.find_nearest_neighbor(patch, Np) for patch in Tp] for Np in Nps]
		entropy_est()

	except ValueError:
		print("incorrect command line argument formatting", args)
		sys.exit(-1)