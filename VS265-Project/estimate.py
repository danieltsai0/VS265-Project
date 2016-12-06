import pickle, signal, sys, util, config
import numpy as np
import matplotlib.pyplot as plt


"""
Estimates entropy and relative dimension of a given dataset. 

Arguments: 
	n_neighbors_pow: used to exponentiate the number dos
	tod: type(s) of data being measured 

Example calls:
	python main.py estimate 17 nat ica
	python main.py estimate 17 nat ica pca

	Note that the order of the tod doesn't matter
"""
def estimate(*args):

	try:
		_, _, n_neighbors_pow, *tod = args
		estimates, nnp = [], int(n_neighbors_pow)

		for d in tod:

			if d not in config.types_of_data:
				print("Unknown type of data requested. Skipping entropy calculation for", d)
				continue

			# Load patches 
			Tp, Nps = util.load_patches(d, nnp)
			# Generate distances for each target patch
			D = [[util.find_nearest_neighbor(patch, Np) for patch in Tp] for Np in Nps]
			estimates.append( entropy_est() )

		pickle.dump(estimates, config.gen_data_dir 
							   + n_neighbors_pow
							   + config.entropy_pickle_suffix)

	except ValueError:
		print("incorrect command line argument formatting", args)
		sys.exit(-1)