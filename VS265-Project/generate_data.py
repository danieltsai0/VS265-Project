import pickle, util, sys
import numpy as np
import matplotlib.pyplot as plt

from util import *

"""
Generate 5 types of data. 
Each image is 1024 x 1024, stored as a numpy matrix. 
Each dataset is an array of numpy matrices.
Serialize dataset with pickle.

Types of data:
	Gaussian, 1/f, 1/f^2, spectrum-equalized noise, natural scenes

Arguments: 
	dataset_fn - natural image dataset
	fn - filename to save serialized data to, prepended with type of data generated
	R - generates R-by-R images
	n - generate n-by-n patches from the images
"""

def generate_data(dataset_fn, fn, R, n):

	# generate gaussian
	# generate 1/f
	# generate 1/f^2
	# generate spectrum-equalized
	# generate natural images
	return

"""
Generate patches for a given dataset. Assumes that each image is a 1024x1024 numpy matrix.

Arguments: 
	dataset_fn 

Returns:
	tuple of patches for T and N respectively
"""
def generate_patches(dataset_fn, R, n):
	# split dataset into two
	# generate patches from each image in each set
	with open(dataset_fn, "rb") as dataset:
		data = pickle.load(dataset)

	randomized = np.random.permutation(data)
	Ti, Ni = randomized[:len(randomized)//2], randomized[len(randomized)//2:]
	return get_patches(Ti, R, n), get_patches(Ni, R, n)

"""
Generate patches for a given imagset. Assumes that each image is a R-by-R numpy matrix.

Arguments:
	imageset - array of numpy matrices
	n - size of patches to generate (n-by-n)

Returns: 
	array of n-by-n numpy matrices
"""
def get_patches(imageset, R, n):
	
	ret, r = [], np.delete(np.arange(0, R, n), -1, 0)

	for image in imageset:
	 	ret.extend([np.array(image[x:x+n,y:y+n]) for x in r for y in r])

	return ret