import array, sys, os, math
import numpy as np

##########
# Macros #
##########


types_of_data = ["gaussian", "1/f", "1/f2", "equalized", "natural"]
n_neighbors = []
data_pickle_suffix = "_data.pickle"
patches_pickle_suffix = "_patches.pickle"
natural_image_dir = "..\\Data\\vanhateren_imc"
gamma = 0.577215665
num = 150


###########
# Methods #
###########

"""
Loads a numpy dataset and return it.
"""
def load_dataset(tod, n_neighbors):
	return np.load(filename)

def load_patches(tod):
	raise NotImplementedError 

def find_nearest_neighbor(patch, Np):
	return np.min([np.linalg.norm(patch - npatch) for npatch in Np])


"""
Estimates the entropy of a given set of distances.

Arguments: 
	k - dimension of _________________________________
	d - array of arrays (each representing a different N)

Returns: 
	array of estimated entropy of target patches for each N
"""
def entropy_est(k, d):
	A_div_k = k * math.pi**(k/2) / func_Gamma(k/2 + 1)
	res = k*d + math.log(A_div_k, 2) + logN + gamma/math.log(2)


"""
Euclidian Gamma function (i think...)
"""
def func_Gamma(v):
	raise NotImplementedError


"""
Generates 8bit square Gaussian distributed noise

Arguments: 
	var - variance of gaussian distribution
	R - size of image (RxR)
	num - number of images to generate

Returns: 
	array of numpy matrices
"""
def generate_gaussian(var, R, num=1):
	raise NotImplementedError


"""
Generates 8bit square pink noise

Arguments: 
	var - variance of gaussian distribution
	f_pow - standard deviation ~ 1/f**f_pow
	R - size of image (RxR)
	num - number of images to generate

Returns: 
	array of numpy matrices
"""
def generate_pink(var, f_pow, R, num=1):
	raise NotImplementedError


"""
Generates 8bit square spectrum-equalized noise

Arguments: 
	R - size of image (RxR)
	num - number of images to generate

Returns: 
	array of numpy matrices
"""
def generate_equalized(R, num=1):
	raise NotImplementedError

"""
Crops natural images to RxR from image numbers listed 
in filenums.txt. 

Arguments: 
	nat_image_dir - directory containing natural images
	R - size (RxR) to crop to

Returns: 
	(array of numpy matrices, sample variance of natural images used)
"""
def crop_natural_images(nat_image_dir, R):

	raise NotImplementedError

	try:
		for filename in os.listdir(nat_image_dir):
			fin = open( filename, 'rb' )
			s = fin.read()
			fin.close()
			arr = array.array('H', s)
			arr.byteswap()
			img = numpy.array(arr, dtype='uint16').reshape(1024,1536)

	except FileNotFoundError as e:
		print ("Directory not found error({0}): {1}".format(e.errno, e.strerror))
		return -1

	except IOError as e:
		print ("I/O error({0}): {1}".format(e.errno, e.strerror))

	except:
		print("Unexpected error encounteredc", sys.exc_info()[0])
