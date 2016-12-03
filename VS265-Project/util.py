import array, sys, os, math
import numpy as np


##########
# Macros #
##########


types = ["Gaussian", "1/f", "1/f^2", "spectrum-equalized", "natural"]
natural_image_dir = "..\\Data\\vanhateren_imc"
gamma = 0.577215665


###########
# Methods #
###########

"""
Loads a numpy dataset and return it.
"""
def load_dataset(filename):
	return np.load(filename)

def find_nearest_neighbor(patch, Np):
	return np.min([np.linalg.norm(patch - npatch) for npatch in Np])

def entropy_est(k, d, logN):
	A_div_k = k * math.pi**(k/2) / func_Gamma(k/2 + 1)
	res = k*d + math.log(A_div_k, 2) + logN + gamma/math.log(2)

def func_Gamma(v):
	return -1

def generate_gaussian(var):
	return

def generate_pink(var, f_pow):
	return

def generate_equalized():
	return

def crop_natural_images(nat_image_dir):

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


	