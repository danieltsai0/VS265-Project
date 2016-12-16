import array, sys, os, math, config, pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import FastICA, PCA

###########
# Methods #
###########

"""
Provides view of 2D numpy matrix in greyscale
"""
def view(mat):
	plt.matshow(mat, cmap=plt.cm.gray)
	plt.show()
	return

"""
Loads a numpy dataset and return it.
"""
def load_dataset(tod):
	with open(config.gen_data_dir+tod+config.data_pickle_suffix, "rb") as dataset:
		data = pickle.load(dataset)

	return data

def load_patches(tod):
	with open(config.gen_data_dir+tod+config.patches_pickle_suffix, "rb") as patchset:
		patches = pickle.load(patchset)
		Tp, Np = patches["Tp"], patches["Np"]

	return Tp, Np

def find_nearest_neighbor(patch, Np):
	return np.min([np.linalg.norm(patch - npatch) for npatch in Np])

def log(x):
	if x == 0:
		return 0
	else:
		return math.log(x, 2)

"""
Estimates the entropy of a given set of distances.

Arguments: 
	k - dimension of _________________________________
	d - array of arrays (each representing a different N)

Returns: 
	array of estimated entropy of target patches for each N
"""	
def entropy_est(k, d, N):
	A_div_k = math.pi**(k/2) / math.gamma(k/2 + 1)
	res = k*d + math.log(A_div_k * N, 2) + config.gamma/math.log(2)
	return res


"""
Generates 8bit square Gaussian distributed noise

Arguments: 
	var - variance of gaussian distribution
	R - size of image (RxR)
	num - number of images to generate

Returns: 
	array of numpy matrices
"""
def generate_gaussian(mean, var, R, num):
	noise_ims = []
	for i in range(num):
		noise_im = np.zeros((R,R))
		for j in range(R):
			for k in range(R):
				smpl = np.random.normal(mean, math.sqrt(var))
				noise_im[j][k] = math.floor(smpl + 0.5) #discretize
		noise_ims.append(noise_im)
	return noise_ims


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
	(numpy array of numpy matrices, sample variance of natural images used)
"""
def crop_natural_images(nat_image_dir, R):
	
	try:
		# List to contain cropped images
		imgs = []

		filenums = pickle.load(open(config.filenums_fn, "rb"))
		for filename in os.listdir(nat_image_dir):

			filenum = filename.replace("k", " ").replace(".", " ").split()[1]
			
			if filenum not in filenums:
				continue
			# Load image file
			fin = open(nat_image_dir+filename, 'rb' )
			s = fin.read()
			fin.close()
			# Create uint16 array for data
			arr = array.array('H', s)
			arr.byteswap()
			# Convert to uint8 data and crop
			img = np.array(arr, dtype='uint16').reshape(1024,1536)[:,256:1280]
			img = img*(255/np.max(img))
			#img = img/257


			imgs.append(img)

		return np.array(imgs)

	except FileNotFoundError as e:
		print ("Directory not found error({0}): {1}".format(e.errno, e.strerror))
		return -1

	except IOError as e:
		print ("I/O error({0}): {1}".format(e.errno, e.strerror))

	except:
		print("Unexpected error encountered", sys.exc_info()[0])


def reconstruction(components, dataset):
	raise NotImplementedError

"""
takes array of cropped natural images (RxR);
returns array of ICA reconstruction for each image
based on n components

if n is unspecified, all are used

"""
def ica_reconst(images, R, n=None):
    #reshape image data into vectors
    num_ims, im_size = images.shape[0], images[0].size
    im_vecs = (images.reshape(num_ims, im_size)).T #shape(im_size,num_ims)

    #train ica
    ica = FastICA(n_components=n)
    basis = ica.fit_transform(im_vecs) #shape(im_size, n)


    #project and reconstruct
    #im_weights = im_vecs @ basis #(num_ims, n)
    im_weights = ica.mixing_ #(num_ims, n)
    reconst = im_weights @ basis.T #(num_ims, im_size)
    reconst.reshape(images.shape)
    for i in range(len(reconst)):
        im = reconst[i]
        im += abs(np.min(im))
        reconst[i] = im*(255/np.max(im))
    #reconst = np.floor( reconst + abs(reconst.min()) )#rescale

    return reconst.reshape(images.shape)


"""
takes array of cropped natural images (RxR);
returns array of PCA reconstruction for each image
based on n components

if n is unspecified, all are used

"""
def pca_reconst(images, R, n=None):
    #reshape image data into vectors
    num_ims, im_size = images.shape[0], images[0].size
    im_vecs = (images.reshape(num_ims, im_size)).T

    #train ica
    pca = PCA(n_components=n)
    basis = pca.fit_transform(im_vecs)
    
    #project and reconstruct
    #im_weights = pca.mixing_ #(num_ims, n)
    im_weights = im_vecs.T @ basis #(num_ims, n)
    reconst = im_weights @ basis.T #(num_ims, im_size)
    
    # reconst = reconst/reconst.size #normalize
    # reconst = np.floor( reconst + abs(reconst.min()) )#rescale
    for i in range(len(reconst)):
        im = reconst[i]
        im += abs(np.min(im))
        reconst[i] = im*(255/np.max(im))

    return reconst.reshape(images.shape)
