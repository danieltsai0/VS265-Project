import pickle, signal, sys, util
import numpy as np
import matplotlib.pyplot as plt

# objects to be pickled: 
# 	learned model
# 	iterator over dataset


"""
Estimates entropy and relative dimension of a given dataset. 

Arguments: 
	tod: type of data being measured 
	n_neighbors: number of neighbors to compare each patch to
"""
def estimate(*args):

	if len(args) != 3:
		print("Error in input")
		sys.exit(-1)

	_, _, tod, n_neighbors = args

	if dataset_fn == "" or n_neighbors == -1: 
		print("Error in input")
		sys.exit(-1)

	# Load patches from dataset
	Tp, Np = util.load_patches(tod, n_neighbors)
	# Generate distances for each target patch
	D = [util.find_nearest_neighbor(patch, Np) for patch in Tp]


if __name__ == "__main__":

	main(sys.argv)
