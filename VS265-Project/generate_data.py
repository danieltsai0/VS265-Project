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
	R - generates R-by-R images
"""

def generate_data(*args):
	
	varG, varP, data_ary = None, None, []
	_, _, R = args

	# generate gaussian
	data_ary.append(util.generate_gaussian(varG, R, util.num))
	# generate 1/f
	data_ary.append(util.generate_pink(varP, 1, R, util.num))
	# generate 1/f^2
	data_ary.append(util.generate_pink(varP, 2, R, util.num))
	# generate spectrum-equalized
	data_ary.append(util.generate_equalized(R, util.num))
	# generate natural images
	data_ary.append(util.crop_natural_images(natural_image_dir, R))
	
	for tod, data in zip(util.types_of_data, data_ary):
		pickle.dump(data, tod+util.pickle_suffix)

"""
Generate patches for a given dataset. Assumes that each image is a 1024x1024 numpy matrix.

Arguments: 
	dataset_fn 
	R
	n

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