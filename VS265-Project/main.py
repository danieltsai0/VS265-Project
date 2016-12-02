import pickle, signal, sys, util
import numpy as np
import matplotlib.pyplot as plt

# objects to be pickled: 
# 	learned model
# 	iterator over dataset


"""
Main method that either loads or creates a new model that is to be 
trained. 

Arguments: 
	tod: type of data being measured 
	n_neighbors: number of neighbors to compare each patch to
"""
def main(*args):

	if len(args) > 3:
		print("Error in input")
		sys.exit(-1)

	_, tod, n_neighbors = args

	if dataset_fn == "" or n_neighbors == -1: 
		print("Error in input")
		sys.exit(-1)
		
	Tp, Np = load (from_data_patches)
	for patch in Tp:
		D = util.find_nearest_neighbor(patch, Np)


if __name__ == "__main__":

	if len(sys.argv) > 1:
		main(sys.argv)
	else:
		main()