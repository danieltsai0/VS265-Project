import pickle, signal, sys, util
import numpy as np
import matplotlib.pyplot as plt

"""
Estimates entropy and relative dimension of a given dataset. 

Arguments: 
	tod: type of data being measured 
	------n_neighbors: number of neighbors to compare each patch to------------------
"""
def estimate(*args):

	if len(args) != 3:
		print("Error in input")
		sys.exit(-1)

	_, _, tod, n_neighbors = args

	if dataset_fn == "" or n_neighbors == -1: 
		print("Error in input")
		sys.exit(-1)

	# Load patches 
	Tp, Nps = util.load_patches(tod)
	# Generate distances for each target patch
	D = [[util.find_nearest_neighbor(patch, Np) for patch in Tp] for Np in Nps]
	entropy_est()
